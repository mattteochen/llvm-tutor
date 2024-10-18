//=============================================================================
// FILE:
//    OmpDependencyFinder.cpp
//
// DESCRIPTION:
//    Find the static computational graph of a OpenMP task program
//
// USAGE:
//    New PM
//      opt -load-pass-plugin=libOmpDependencyFinder.so
//      -passes="module-pass-and-analysis"
//      `\`
//        -disable-output <input-llvm-file>
//
//
// License: MIT
//=============================================================================
#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/InlineCost.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

//-----------------------------------------------------------------------------
// OmpDependencyFinder implementation
//-----------------------------------------------------------------------------
// No need to expose the internals of the pass to the outside world - keep
// everything in an anonymous namespace.
namespace {

const std::string targetOmpTaskAllocName = "__kmpc_omp_task_alloc";
const std::string targetOmpTaskSubmissionDepsName = "__kmpc_omp_task_with_deps";
const std::string targetOmpTaskSubmissionName = "__kmpc_omp_task";
const std::string targetOmpDepsStructName = "struct.kmp_depend_info";
const std::string targetOmpAnonStructName = "struct.anon";
const std::string targetOmpOutlinedFnNamePrefix = ".omp_outlined";

constexpr uint8_t OPEN_FLAG = 0;
constexpr uint8_t CLOSE_FLAG_WITH_DEPS = 1;
constexpr uint8_t CLOSE_FLAG = 2;

void visitor(Function &F, std::vector<unsigned> const *entryPointInputs,
             bool clearDeps = true);

// key: fn ID
// value.first: in_deps
// value.second: out_deps
std::map<std::string, std::pair<std::set<unsigned>, std::set<unsigned>>>
    dependentTasks;

// A map from the function name to its Function reference
std::unordered_map<std::string, Function *> fnCache;

// A map from a task entry ID to its input list
std::unordered_map<unsigned, std::vector<unsigned>> taskEntryInputs;

// A set containing free tasks with zero dependencies
std::unordered_map<unsigned, unsigned> freeTasks;

// Taken from
// https://vscode.dev/github/llvm/llvm-project/blob/main/openmp/runtime/src/kmp.h#L2505
const std::unordered_set<uint8_t> outDepsFlags = {2, 3};

// Taken from
// https://vscode.dev/github/llvm/llvm-project/blob/main/openmp/runtime/src/kmp.h#L2505
const std::unordered_set<uint8_t> inDepsFlags = {1};

/// Retrieve the first available numerical label (IR proxy) after the given char
/// @param str The input string referring to a @ref llvm::Instruction
/// @param c The starting point char inside the input string
std::string retrieveLabelFromChar(std::string const &str, const char c) {
  std::string parsed = "";
  unsigned i = 0;
  while (i < str.size() && str[i] != c) {
    i++;
  }
  i++;
  while (i < str.size() && str[i] >= '0' && str[i] <= '9') {
    parsed += str[i++];
  }
  return parsed;
}

/// Retrieve the proxy label which is unnamed from an instruction string. This
/// will retrieve effectively the first proxy being encountered inside the
/// instruction string
/// @param str A string representing a @ref llvm::Instruction
std::string parseUnamedNameFromInstruction(std::string const &str) {
  return retrieveLabelFromChar(str, '%');
};

/// Retrieve the proxy label of a `ptrtoint` source
/// @note This function does not check that the provided instruction is
/// effectively a `ptrtoint` instruction
/// @param str A string representing a @ref llvm::Instruction
std::string parsePtrToIntSource(std::string const &str) {
  unsigned i = 0;
  while (i < str.size() && str[i] != '%') {
    i++;
  }
  return retrieveLabelFromChar(str.substr(i + 1), '%');
};

/// Remove any whitespaces that prefix a string
/// @param str The input string
/// @return The modified string
std::string removeEmptyPrefix(std::string const &str) {
  std::string removed = "";
  unsigned i = 0;
  while (i < str.size() && str[i] == ' ') {
    i++;
  }
  while (i < str.size()) {
    removed += str[i++];
  }
  return removed;
};

/// Create an unique task id identified for task entry lambdas that are shared
/// between different contexts
/// @param id The original task entry id
/// @param unique The unique identifier for the original id
/// @return The unique task entry identifier
std::string uniqueTaskEntryIdentifier(std::string const &id,
                                      const unsigned unique) {
  return id + "#" + std::to_string(unique);
}

/// Recover the original task entry id from an unique generated id. See @ref
/// uniqueTaskEntryIdentifier
/// @param uniqueId The unique task entry id
/// @return The original task id without the unique identifier
unsigned recoverLambdaIdFromUniqueId(std::string const &uniqueId) {
  const auto pos = uniqueId.find('#');
  return static_cast<unsigned>(std::stoi(uniqueId.substr(0, pos)));
}

namespace utils {
template <typename... T>
/// Log util
/// @tparam T A generic message type
/// @param out The desired output stream
/// @param indent The number of spaces to prefix the message content
/// @param noNewLines The number of new lines to postfix the message content
void log(raw_fd_ostream &out, uint8_t indent, uint8_t noNewLines, T &&...args) {
  std::string indentStr = "";
  for (uint8_t i = 0; i < indent; ++i) {
    indentStr += ' ';
  }
  out << indentStr;
  ((out << args << " "), ...); // Fold expression with proper << expansion
  for (uint8_t i = 0; i < noNewLines; ++i) {
    out << "\n";
  }
}
} // namespace utils

/// Check if the instruction is part of a omp task call
/// @param inst A @ref llvm::Instruction
/// @param flag To control open or closing omp task calls (task alloc and task
/// submission)
/// @return The query result
bool isTarget(Instruction const &inst, const uint8_t flag) {
  StringRef instruction_name = inst.getName();
  StringRef opcode_name = inst.getOpcodeName();
  if (auto *CI = dyn_cast<CallInst>(&inst)) {
    StringRef name = CI->getCalledFunction()->getName();
    if (flag == OPEN_FLAG && name.data() == targetOmpTaskAllocName) {
      return true;
    } else if (flag == CLOSE_FLAG_WITH_DEPS &&
               name.data() == targetOmpTaskSubmissionDepsName) {
      return true;
    } else if (flag == CLOSE_FLAG &&
               name.data() == targetOmpTaskSubmissionName) {
      return true;
    }
  }
  return false;
}

/// Retrieve the instructions wrapping a task creation and submission
/// A single basic block may contain multiple task creation instructions
/// @param block A basic block
/// @param includeAllBeforeTaskAllocation A flag to control if to include all
/// instructions or not
/// @return A list of possible instructions containing task creation
std::vector<std::vector<Instruction *>>
taskWithDepsInstructions(BasicBlock &block,
                         bool includeAllBeforeTaskAllocation) {
  bool inside = 0;
  std::vector<Instruction *> instructions;
  std::vector<std::vector<Instruction *>> blockInstructions;
  for (Instruction &inst : block) {
    if ((!inside && isTarget(inst, OPEN_FLAG))) {
      inside = 1;
      instructions.push_back(&inst);
    } else if (inside && (isTarget(inst, CLOSE_FLAG_WITH_DEPS) ||
                          (isTarget(inst, CLOSE_FLAG)))) {
      inside = 0;
      instructions.push_back(&inst);
      blockInstructions.push_back(std::vector(instructions));
      instructions.clear();
    } else if (inside) {
      instructions.push_back(&inst);
    } else if (!inside && includeAllBeforeTaskAllocation) {
      instructions.push_back(&inst);
    }
  }
  return blockInstructions;
}

/// Function to check if the GEP is accessing a specific struct
/// @param GEP A pointer to a @ref llvm::GetElementPtrInst
/// @param structName The target struct name
/// @return A flag stating if the GEP accesses an instance of the target struct
bool GEPIsSpecificStruct(const GetElementPtrInst *GEP,
                         const std::string &structName) {
  // Get the source type of the GEP (the type that is being indexed)
  Type *sourceType = GEP->getSourceElementType();

  if (sourceType->isStructTy()) {
    StructType *structType = cast<StructType>(sourceType);
    const auto structName = structType->getName().str();
    if (structName.find(structName) != std::string::npos) {
      utils::log(errs(), 2, 1, "This GEP is accessing the struct:", structName);
      return true;
    } else {
      utils::log(errs(), 2, 1,
                 "This GEP is accessing an array of the struct different "
                 "from the target struct: ",
                 structType->getName());
      return false;
    }
  }
  return false;
}

/// Function to check if the GEP is accessing an array of a specific struct
/// @param GEP A pointer to a @ref llvm::GetElementPtrInst
/// @param structName The target struct name
/// @return A flag stating if the GEP accesses an array of the target struct
bool GEPIsArrayOfSpecificStruct(const GetElementPtrInst *GEP,
                                const std::string &structName) {
  // Get the source type of the GEP (the type that is being indexed)
  Type *sourceType = GEP->getSourceElementType();

  // Check if the source type is an array
  if (sourceType->isArrayTy()) {
    utils::log(errs(), 2, 1, "The type being indexed is an array");
    // Get the element type of the array
    ArrayType *arrayType = cast<ArrayType>(sourceType);
    Type *elementType = arrayType->getElementType();

    // Check if the element type is a struct
    if (elementType->isStructTy()) {
      StructType *structType = cast<StructType>(elementType);

      // Compare the struct name with the one we are looking for
      if (structType->getName().data() == structName) {
        utils::log(errs(), 2, 1,
                   "This GEP is accessing an array of the struct:", structName);
        return true;
      } else {
        utils::log(errs(), 2, 1,
                   "This GEP is accessing an array of the struct different "
                   "from the target struct: ",
                   structType->getName());
      }
    }
  } else {
    utils::log(errs(), 2, 1, "The type being indexed is not an array");
  }
  return false;
}

/// Find the GEP reference usage and print it
/// @param GEP A pointer to a @ref GetElementPtrInst instance
void GEPUses(const GetElementPtrInst *GEP) {
  std::string stringBuff;
  raw_string_ostream stream(stringBuff);
  GEP->print(stream);
  utils::log(errs(), 2, 1, "Analyzing uses of GEP instruction: ", stringBuff);

  // Iterate through all uses of the GEP instruction
  for (const Use &U : GEP->uses()) {
    const User *user =
        U.getUser(); // Get the user (instruction that uses the GEP result)

    // Check if the GEP result is used in a store instruction
    if (const StoreInst *storeInst = dyn_cast<StoreInst>(user)) {
      utils::log(errs(), 2, 1, "GEP result is being stored in memory");
    }
    // Check if the GEP result is used in a load instruction
    else if (const LoadInst *loadInst = dyn_cast<LoadInst>(user)) {
      utils::log(errs(), 2, 1, "GEP result is being loaded from memory");
    }
    // Check if the GEP result is passed as an argument to a function call
    else if (const CallInst *callInst = dyn_cast<CallInst>(user)) {
      utils::log(errs(), 2, 1, "GEP result is being used in a function call");
    }
    // Handle other types of instructions
    else {
      utils::log(errs(), 2, 1, "Not recognized use");
    }
  }
}

/// Define the store instruction data type
/// Used in @ref GEPStoreUses
enum StoreType { PointerStore, ValueStore };

/// Recover the source pointer casted to int in a `ptrtoint` call.
/// This function does a linear scan over the instructions and finds the
/// associated `ptrtoint` call that matches the provided label
/// @param instructions The instructions list
/// @param label The result proxy where the `ptrtoint` operation is being saved
/// @return The source pointer inside the `ptrtoint` call label
std::string
recoverPtrToIntSource(std::vector<Instruction *> const &instructions,
                      const std::string &label) {
  std::string sourceLabel = "";
  std::string stringBuff;
  raw_string_ostream stream(stringBuff);
  for (const auto *I : instructions) {
    I->print(stream);
    const auto currLabel = parseUnamedNameFromInstruction(stringBuff);

    if (currLabel == label) {
      sourceLabel = parsePtrToIntSource(stringBuff);
      break;
    }
    stringBuff.clear();
  }
  assert(sourceLabel.size() > 0);
  return sourceLabel;
}

/// Retrieve omp dependency struct pointer retrieval instruction actions to
/// detect omp dependencies This function is specialized on IR of the omp task
/// dependency scheduling
/// @tparam T The type of storage we are looking for. @ref
/// StoreType::PointerStore for dependencies sources and @ref
/// StoreType::ValueStore for dependencies types
/// @param GEP A pointer to a get element pointer instance
/// @return A vector containing pairs of the extracted stored proxy label and
/// the store instruction as string itself
template <StoreType T>
std::vector<std::pair<int, std::string>>
GEPStoreUses(const GetElementPtrInst *GEP) {
  // Retrieve the label of a pointer
  auto getPtrLabel = [](const std::string &str) {
    std::string label = "";
    unsigned i = 0;
    // Remove any initial whitespace
    while (i < str.size() && str[i] == ' ') {
      i++;
    }
    while (i < str.size() && str[i] != ' ') {
      if (str[i] >= '0' && str[i] <= '9') {
        label += str[i];
      }
      i++;
    }
    return label;
  };

  // Retrieve the label of a constant value
  auto getValueLabel = [](const std::string &str) {
    std::string label = "";
    unsigned i = 0;
    // Remove any initial whitespace
    while (i < str.size() && str[i] == ' ') {
      i++;
    }
    uint8_t encounteredSpaces = 0;
    // Move pointer to the value being stored
    while (i < str.size() && str[i] != ' ') {
      i++;
    }
    i++;
    // Save our value
    while (i < str.size() && (str[i] >= '0' && str[i] <= '9')) {
      label += str[i];
      i++;
    }
    return label;
  };

  std::vector<std::pair<int, std::string>> res;
  std::string stringBuff;
  raw_string_ostream stream(stringBuff);

  GEP->print(stream);
  utils::log(errs(), 2, 1,
             "Analyzing store uses of GEP instruction: ", stringBuff);
  stringBuff.clear();

  // Iterate through all uses of the GEP instruction
  for (const Use &U : GEP->uses()) {
    const User *user =
        U.getUser(); // Get the user (instruction that uses the GEP result)

    // Check if the GEP result is used in a store instruction. We are interested
    // in the first one (there should be only one use though).
    if (const StoreInst *storeInst = dyn_cast<StoreInst>(user)) {
      // TODO: can this be done in a easier way?
      // Retrieve the label being store
      const Value *storedValue = storeInst->getValueOperand();
      storedValue->print(stream);
      utils::log(errs(), 2, 1, "Stored value: ", stringBuff);

      std::string label;
      if constexpr (T == StoreType::PointerStore) {
        label = getPtrLabel(stringBuff);
      } else {
        label = getValueLabel(stringBuff);
      }
      stringBuff.clear();
      storeInst->print(stream);
      const std::string storeInstructionStr = removeEmptyPrefix(stringBuff);
      res.push_back({std::stoi(label), storeInstructionStr});
      stringBuff.clear();
    }
  }
  return res;
}

/// Retrieve the omp task entry id from its whole name
/// @param name The task entry point lambda name
/// @return The ID associated with the task entry lambda
std::string getOmpTaskEntryId(std::string const &name) {
  unsigned i = 0;
  // Prepend with a zero as the id will be converted to an unsigned. The first
  // created task entry has no numeric id hence we assign it to zero
  // TODO: check if by default clang will generate a task entry with id = 0
  std::string res = "0";
  // We could directly jump to the index as they all have the same prefix. But
  // for safety let's go in this way for now
  while (i < name.size() && (name[i] < '0' || name[i] > '9')) {
    i++;
  }
  while (i < name.size()) {
    res += name[i++];
  }
  return res;
}

/// Find the input labels to a omp task entry lambda
/// @param I The omp task alloc instruction
/// @param swapMap A map containing swap values for input proxy labels
/// @return A @ref std::vector representing the inputs labels
std::vector<unsigned> findTaskEntrypointInputs(
    Instruction &I, std::unordered_map<unsigned, unsigned> const &swapMap) {
  std::vector<unsigned> inputs;
  auto cleanOperandString = [](std::string const &str) {
    unsigned index = 0;
    std::string cleaned = "";
    while (index < str.size() && str[index] != '%') {
      index++;
    }
    index++;
    while (index < str.size() && str[index] >= '0' && str[index] <= '9') {
      cleaned += str[index++];
    }
    return cleaned;
  };

  auto saveStore = [&cleanOperandString, &inputs,
                    &swapMap](auto *storeInst, const unsigned index) {
    std::string stringBuff;
    raw_string_ostream stream(stringBuff);
    storeInst->getOperand(index)->print(stream);
    const auto parsedOperand = cleanOperandString(stringBuff);
    utils::log(errs(), 2, 1,
               "Found a store instruction linked to the omp task "
               "allocation struct. Stored value label:",
               parsedOperand);
    const unsigned numericalValue = std::stoi(parsedOperand);

    // Swap the label if any replacement has been provided
    inputs.push_back(swapMap.find(numericalValue) == swapMap.end()
                         ? numericalValue
                         : swapMap.at(numericalValue));
  };

  // The following block is quite nested but this is needed to retrieve a task
  // entry lambda input parameters The current visits across llvm nodes (users)
  // are defined on a IR produced by clang14
  if (auto *callInst = dyn_cast<CallInst>(&I)) {
    // Check if it's a call to __kmpc_omp_task_alloc
    assert(callInst->getCalledFunction()->getName() == targetOmpTaskAllocName);
    Value *allocResult = callInst;

    for (auto *user : allocResult->users()) {
      if (auto *bitCastInst = dyn_cast<BitCastInst>(user)) {
        Value *bitCastedValue = bitCastInst;

        for (auto *user2 : bitCastedValue->users()) {
          if (auto *loadInst = dyn_cast<LoadInst>(user2)) {
            Value *loadedValue = loadInst;

            // One input data case
            // If there are multiple input data, this pattern is not valid
            // (clang14 IR)
            for (auto *user3 : loadedValue->users()) {
              if (auto *storeInst = dyn_cast<StoreInst>(user3)) {
                // Parse only the fist operand as it contains the reference we
                // are looking for
                saveStore(storeInst, 0);
              }
            }

            // Multiple input data case: first input data has a different
            // handling wrt to the subsequent ones (clang14 IR)
            for (auto *user3 : loadedValue->users()) {
              if (auto *gepInst = dyn_cast<GetElementPtrInst>(user3)) {
                // This block retrieves the inputs from the second one
                for (auto *user4 : gepInst->users()) {
                  if (auto *bitCastInst2 = dyn_cast<BitCastInst>(user4)) {
                    for (auto *user5 : bitCastInst2->users()) {
                      if (auto *storeInst = dyn_cast<StoreInst>(user5)) {
                        // Parse only the fist operand as it contains
                        // the reference we are looking for
                        saveStore(storeInst, 0);
                      }
                    }
                  }
                }
              } else if (auto *bitCastInst3 = dyn_cast<BitCastInst>(user3)) {
                // This block retrieves the first input
                for (auto *user6 : bitCastInst3->users()) {
                  if (auto *storeInst = dyn_cast<StoreInst>(user6)) {
                    // Parse only the fist operand as it contains the
                    // reference we are looking for
                    saveStore(storeInst, 0);
                  }
                }
              }
            }
            break;
          }
        }
        break;
      }
    }
  }
  return inputs;
}

/// Visit and find dependencies, inputs of children tasks (tasks created inside
/// a task)
/// @param inputs The children input list
/// @param entryPointName The children task entry point lambda name
void ompDependencyFinderChilds(std::vector<unsigned> const &inputs,
                               std::string const &entryPointName) {
  // Recover the cached function
  Function *fn = fnCache[entryPointName];
  assert(fn);
  utils::log(errs(), 2, 1, "Recovered child fn", fn->getName());

  // Log
  utils::log(errs(), 2, 1, "Inputs to the entry point:");
  utils::log(errs(), 2, 0, "[");
  for (auto input : inputs) {
    const std::string formattedInput = "%" + std::to_string(input);
    utils::log(errs(), 2, 0, formattedInput);
  }
  utils::log(errs(), 2, 1, "]");

  // Recursively visit this task entry point
  visitor(*fn, &inputs, false);
}

/// Compute all the GEP instructions to the target dependency struct by
/// extracting dependency sources and types
/// @param blockInstructions The instructions of a function block
/// @param inputMap A label swap map
/// @return The extracted informations as mapping
std::unordered_map<std::string, std::pair<unsigned, unsigned>>
storeGEPDepStructAccesses(
    std::vector<Instruction *> const &blockInstructions,
    std::unordered_map<unsigned, unsigned> const &inputMap) {
  std::unordered_map<std::string, std::pair<unsigned, unsigned>>
      precomputedDeps;
  std::vector<unsigned> foundDepsLabels;
  std::vector<std::string> depsStoreKeys;

  // The following block relies on the fact that gep accesses to the same omp
  // deps struct are ordered by field
  for (auto &I : blockInstructions) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      if (GEPIsArrayOfSpecificStruct(GEP, targetOmpDepsStructName)) {
        // This block retrieves the dependency sources and ther types (in or
        // out)

        // KAIXI: Do refer to technical docs for these accesses
        auto lastIndex = GEP->getNumOperands();
        // We do expect structs with more than one field
#ifdef ENABLE_ASSERTIONS
        assert(lastIndex > 0);
#endif
        lastIndex--;

        Value *value = GEP->getOperand(lastIndex);
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(value)) {
          const auto value = CI->getValue();
          if (value == 0) {
            utils::log(errs(), 2, 1,
                       "GEP instruction on the dependency source field");
            const auto uses = GEPStoreUses<StoreType::PointerStore>(GEP);
            for (auto &use : uses) {
              int label = use.first;
              depsStoreKeys.push_back(use.second);
              // The label being stored in the GEP reference will be the output
              // alias from a `ptrtoint` instruction. The real data source is
              // available inside the `ptrtoint` call which is before this
              // instruction. Recover it.
              label = std::stoi(recoverPtrToIntSource(blockInstructions,
                                                      std::to_string(label)));
              auto foundDepsLabel = label;
              // Map the label to a parent source if available
              if (inputMap.find(foundDepsLabel) != inputMap.end()) {
                foundDepsLabel = inputMap.at(foundDepsLabel);
              }
              foundDepsLabels.push_back(foundDepsLabel);
            }
          } else if (value == 2) {
            utils::log(errs(), 2, 1,
                       "GEP instruction on the dependency type field");
            const auto uses = GEPStoreUses<StoreType::ValueStore>(GEP);
#ifdef ENABLE_ASSERTIONS
            assert(foundDepsLabels.size());
            assert(foundDepsLabels.size() == uses.size() &&
                   uses.size() == depsStoreKeys.size());
#endif
            const unsigned size = uses.size();
            for (unsigned i = 0; i < size; ++i) {
              const int depType = uses[i].first;
              precomputedDeps[depsStoreKeys[i]] = {foundDepsLabels[i], depType};
            }
            // GEPs in this current struct are sorted by field, reset the
            // buffers for the next struct
            foundDepsLabels.clear();
            depsStoreKeys.clear();
          }
        }
      }
    }
  }
  return precomputedDeps;
}

/// Find omp dependencies from a list of instructions containing a sequence
/// of omp task calls. Dependencies are discovered by leveraging GEP calls
/// which populates omp deps structs. Other instructions might be present
/// inside the input sequence and are filtered out. This pass will also save
/// for each omp task entry lambda the relative inputs.
/// GEP accesses to the resolved dependencies structs (with dependency sources
/// and types) are provided as input.
/// @param instructions A @ref std::vector of @ref Instruction
/// @param allFnInstruction The complete @ref llvm::Instruction list of the
/// current visited function
/// @param targetStructName The struct name to search
/// @param entryPointInputs A vector of the current task entry lambda inputs
/// @param precomputedDeps A map between dependencies store instructions to the
/// @param taskEntryPointCount A counting map from task entry name to its
/// frequency in the current @ref llvm::Function block relative dependency
/// source and type labels
template <bool ShowInbounds = false>
void ompDependenciesFinder(
    std::vector<Instruction *> const &instructions,
    std::vector<Instruction *> const &allFnInstructions,
    const std::string &targetStructName,
    std::vector<unsigned> const *entryPointInputs,
    std::unordered_map<std::string, std::pair<unsigned, unsigned>> const
        &precomputedDeps,
    std::unordered_map<std::string, unsigned> const &taskEntryPointCount) {
  std::unordered_map<unsigned, unsigned> inputMap;

  std::string ompTaskEntryLambdaName = "";
  std::string stringBuff = "";
  raw_string_ostream stream(stringBuff);

  utils::log(errs(), 2, 1, "\nFirst instruction:", *(instructions[0]));
  utils::log(errs(), 2, 1,
             "Last instruction:", *(instructions[instructions.size() - 1]));

  for (auto &inst : instructions) {
    if (CallInst *ompTaskAPICall = dyn_cast<CallInst>(inst)) {
      auto calledFnName = ompTaskAPICall->getCalledFunction()->getName().str();
      // It's not either a task creation open call or close call
      if ((calledFnName != targetOmpTaskAllocName) &&
          (calledFnName != targetOmpTaskSubmissionDepsName) &&
          (calledFnName != targetOmpTaskSubmissionName)) {
      } else if (calledFnName == targetOmpTaskAllocName) {
        // Extract the executed lambda reference
        const auto noOperands = ompTaskAPICall->getNumOperands();
        // The task entry lambda arguments is stored in the -2 index
        Value *ompTaskEntryLambdaArg =
            ompTaskAPICall->getArgOperand(noOperands - 2);
        utils::log(errs(), 2, 1,
                   "Lambda called name:", ompTaskEntryLambdaArg->getName());
        ompTaskEntryLambdaName = ompTaskEntryLambdaArg->getName().str();

        inst->print(stream);
        const std::string calledInstruction = stringBuff;
        stringBuff.clear();
        utils::log(errs(), 2, 1, "From instruction:", calledInstruction);

        // Find the task inputs. Dependencies data will figure in this
        // input vector as they must be shared variables from a parallel region
        taskEntryInputs[std::stoi(getOmpTaskEntryId(ompTaskEntryLambdaName))] =
            findTaskEntrypointInputs(*inst, inputMap);
      } else if (calledFnName == targetOmpTaskSubmissionName) {
        // When encountering a closing API call, if it's a task with no
        // deps save it
        freeTasks[(std::stoi(getOmpTaskEntryId(ompTaskEntryLambdaName)))]++;
        // The inputs will have the original labels defined inside the parallel
        // region entry point (omp_outlined) to have a consistent global
        // identification
        ompDependencyFinderChilds(taskEntryInputs[std::stoi(getOmpTaskEntryId(
                                      ompTaskEntryLambdaName))],
                                  ompTaskEntryLambdaName);
      } else if (calledFnName == targetOmpTaskSubmissionDepsName) {
        // The inputs will have the original labels defined inside the parallel
        // region entry point (omp_outlined) to have a consistent global
        // identification
        ompDependencyFinderChilds(taskEntryInputs[std::stoi(getOmpTaskEntryId(
                                      ompTaskEntryLambdaName))],
                                  ompTaskEntryLambdaName);
      }
    } else if (auto *storeInst = dyn_cast<StoreInst>(inst)) {
      storeInst->print(stream);
      const std::string storeInstStr = removeEmptyPrefix(stringBuff);
      if (precomputedDeps.find(storeInstStr) != precomputedDeps.end()) {
        utils::log(errs(), 2, 1,
                   "Found a precomputed dependency instruction. Assigning to "
                   "the current task entry");
#ifdef ENABLE_ASSERTIONS
        assert(precomputedDeps.find(storeInstStr) != precomputedDeps.end());
#endif
        const unsigned depType = precomputedDeps.at(storeInstStr).second;
        unsigned depSource = precomputedDeps.at(storeInstStr).first;
        if (inputMap.find(depSource) != inputMap.end()) {
          depSource = inputMap[depSource];
        }
        // As different task entry lambdas can be called on different
        // dependencies we have to create an unique lambda identifier
        // In case the lambda is called only once we retain the original id
        std::string lambdaId;
        if (taskEntryPointCount.find(ompTaskEntryLambdaName) !=
                taskEntryPointCount.end() &&
            taskEntryPointCount.at(ompTaskEntryLambdaName) > 1) {
          lambdaId = uniqueTaskEntryIdentifier(
              getOmpTaskEntryId(ompTaskEntryLambdaName), depSource);
        } else {
          lambdaId = getOmpTaskEntryId(ompTaskEntryLambdaName);
        }
        // Dispatch input deps and output deps
        if (inDepsFlags.find(depType) != inDepsFlags.end()) {
          utils::log(errs(), 2, 1, "Assigning dependent tasks. Holder:",
                     ompTaskEntryLambdaName, " input dep with:", depSource);
          dependentTasks[lambdaId].first.insert(depSource);
        } else if (outDepsFlags.find(depType) != outDepsFlags.end()) {
          utils::log(errs(), 2, 1, "Assigning dependent tasks. Holder:",
                     ompTaskEntryLambdaName, " output dep with:", depSource);
          dependentTasks[lambdaId].second.insert(depSource);
        }
      }
      stringBuff.clear();
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(inst)) {
      if (GEPIsSpecificStruct(GEP, targetOmpAnonStructName)) {
        // This is needed to retrieve the inputs loaded from an anon struct
        // These structs contain the inputs for the current task entry point.
        // Their original reference are stored inside entryPointInputs which is
        // a list of inputs for the current task entry point but filled at the
        // parent level
        //
        // An example:
        //
        // %1 = task_alloc(..., task_entry_X)
        // # compiler stores inside %1 the allocated struct inputs
        // # The previous inputs are captured at previous stages of this pass an
        // forwarded here inside the `entryPointInputs` variable
        //
        // void task_entry_X():
        //    %1 = struct.anon
        //    %2 = load type, struct.anon %1 <-- we have to match this proxy
        //    with its original proxy to have global identifiers
        utils::log(errs(), 2, 1,
                   "Found anon struct. #uses:", GEP->getNumUses());
        unsigned loads = 0;
        for (const auto *anonUser : GEP->users()) {
          anonUser->print(stream);
          utils::log(errs(), 2, 1, "Anon struct ref used in:", stringBuff);
          stringBuff.clear();
          if (auto *loadInst = dyn_cast<LoadInst>(anonUser)) {
            loadInst->print(stream);
            const auto parsed = parseUnamedNameFromInstruction(stringBuff);
            stringBuff.clear();
            utils::log(errs(), 2, 1,
                       "The previous use is a target load instruction relative "
                       "to the task entry inputs. Value "
                       "being saved in proxy:",
                       parsed);

            // If the current task entry lambda has `load` from a `struct.anon`
            // it must have inputs
            assert(entryPointInputs);
            assert(loads < (*entryPointInputs).size());
            // Map the parent label to the current one
            inputMap[std::stoi(parsed)] = (*entryPointInputs)[loads++];
          }
        }
      }
    }
  }
}

/// Print to standard error the parallel task region execution order (graph)
void printDeps() {
  utils::log(errs(), 1, 1, "\nDependency graph:");
  std::unordered_map<unsigned, std::vector<std::string>> depsCount;
  for (auto &kv : dependentTasks) {
    const std::string formattedLambdaName = 'L' + (kv.first);
    utils::log(errs(), 0, 1, "\n\nLambda name id:", formattedLambdaName);
    if (taskEntryInputs[recoverLambdaIdFromUniqueId(kv.first)].size()) {
      utils::log(errs(), 0, 1, "Inputs:");
      utils::log(errs(), 0, 0, "[");
      for (auto &input :
           taskEntryInputs[recoverLambdaIdFromUniqueId(kv.first)]) {
        const std::string formattedInput = '%' + std::to_string(input);
        utils::log(errs(), 0, 0, formattedInput);
      }
      utils::log(errs(), 0, 1, "]");
    }
    utils::log(errs(), 0, 0, "Input dependencies:");
    for (auto &d : kv.second.first) {
      const std::string formatted = '%' + std::to_string(d);
      utils::log(errs(), 0, 0, formatted);
    }
    std::set<std::string> uniqueParents;
    for (auto &d : kv.second.first) {
      for (auto &parent : depsCount[d]) {
        uniqueParents.insert(parent);
      }
    }
    if (uniqueParents.size()) {
      utils::log(errs(), 0, 1,
                 "\nThis task depends on the termination of the "
                 "following parents:");
      utils::log(errs(), 0, 0, "[");
      for (auto &parent : uniqueParents) {
        const std::string formattedLambdaName = 'L' + (parent);
        utils::log(errs(), 0, 0, formattedLambdaName);
      }
      utils::log(errs(), 0, 1, "]");
    } else {
      utils::log(errs(), 0, 1, " ");
    }

    utils::log(errs(), 0, 0, "Out dependencies:");
    for (auto &d : kv.second.second) {
      depsCount[d].push_back(kv.first);
      const std::string formatted = '%' + std::to_string(d);
      utils::log(errs(), 0, 0, formatted);
    }
  }

  utils::log(errs(), 0, 1, "\n\nFree tasks:");
  utils::log(errs(), 0, 0, "[");
  for (auto &task : freeTasks) {
    const std::string formattedLambdaName = 'L' + std::to_string(task.first);
    utils::log(errs(), 0, 0, formattedLambdaName, "x", task.second);
    utils::log(errs(), 0, 0, "(Inputs:");
    utils::log(errs(), 0, 0, "[");
    for (auto &input : taskEntryInputs[task.first]) {
      const std::string formattedInput = '%' + std::to_string(input);
      utils::log(errs(), 0, 0, formattedInput);
    }
    utils::log(errs(), 0, 0, "])");
  }
  utils::log(errs(), 0, 1, "]");
}

/// Find for each task entry of a @ref llvm::Function block its usage count
/// frequency
/// @param instructions A list of @ref llvm::Instruction
/// @return The frequency count as mapping
std::unordered_map<std::string, unsigned>
ompTaskEntryPointUsesCount(std::vector<Instruction *> const &instructions) {
  std::unordered_map<std::string, unsigned> counts;
  for (auto &I : instructions) {
    if (CallInst *ompTaskAPICall = dyn_cast<CallInst>(I)) {
      auto calledFnName = ompTaskAPICall->getCalledFunction()->getName().str();
      // It's not either a task creation open call or close call
      if (calledFnName == targetOmpTaskAllocName) {
        // Extract the executed lambda reference
        const auto noOperands = ompTaskAPICall->getNumOperands();
        // The task entry lambda arguments is stored in the -2 index
        Value *ompTaskEntryLambdaArg =
            ompTaskAPICall->getArgOperand(noOperands - 2);
        utils::log(errs(), 2, 1,
                   "Lambda called name:", ompTaskEntryLambdaArg->getName());
        counts[ompTaskEntryLambdaArg->getName().str()]++;
      }
    }
  }
  return counts;
}

/// This method implements what the pass does
/// @param F The @ref llvm:Function to visit
/// @param entryPointInputs The current task entry point lambda inputs arguments
/// with labels referring to those present in the omp_outlined fn (original
/// labels). Null pointer if this visitor call is called when visiting the
/// omp_outlined itself
/// @param cleanDeps A flag controlling if the known dependency structures have
/// to be cleaned or not for a new parallel task region analysis
void visitor(Function &F, std::vector<unsigned> const *entryPointInputs,
             bool clearDeps) {
  // Clear our dependentTasks holder for now as it will be only printed to
  // standard error
  if (clearDeps) {
    dependentTasks.clear();
    freeTasks.clear();
  }

  // Control the deps log. We log only if this visitor call is not a recursive
  // call but emitted from an omp_outlined (parallel task region entry point)
  const auto logDeps = clearDeps;

  utils::log(errs(), 0, 2,
             "################ START OF FUNCTION ################");
  utils::log(errs(), 0, 1, "Function name:", F.getName(),
             "#args:", F.arg_size());

  // Iterate over basic blocks in the function
  for (BasicBlock &BB : F) {
    utils::log(errs(), 0, 1,
               "\n################# START OF BLOCK ##################");
    // For each basic block retrieves the instructions sequence defining a task
    // creation and submission.  `clearDeps` is true when the visitor is called
    // at omp_outlined level. In this case we don't want to include all the
    // instructions before that task alloc.  In case this visitor is called in a
    // nested call we want to include the informations before the task alloc as
    // they contains data for task entry inputs and dependencies sources.
    const auto parsedInstructions = taskWithDepsInstructions(BB, true);
    utils::log(errs(), 1, 1, "Basic Block task with dependentTasks #:",
               parsedInstructions.size());

    // Recover all the block instructions
    std::vector<Instruction *> allFnInstructions;
    for (auto &i : parsedInstructions) {
      allFnInstructions.insert(allFnInstructions.end(), i.begin(), i.end());
    }

    const auto precomputedDeps =
        storeGEPDepStructAccesses(allFnInstructions, {});
    const auto taskEntryPointCount =
        ompTaskEntryPointUsesCount(allFnInstructions);

    for (auto &i : parsedInstructions) {
      // for (auto &j : i) {
      //   errs() << *j << "\n";
      // }
      ompDependenciesFinder(i, allFnInstructions, targetOmpDepsStructName,
                            entryPointInputs, precomputedDeps,
                            taskEntryPointCount);
    }
    utils::log(errs(), 0, 1,
               "################## END OF BLOCK ###################");
  }

  if (logDeps) {
    printDeps();
  }

  utils::log(errs(), 0, 2,
             "\n################ END OF FUNCTION ##################");
}

// New PM implementation
struct OmpDependencyFinder : PassInfoMixin<OmpDependencyFinder> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    // Cache the Function reference for recursive usage later inside the
    // visitor. Some visitor fn may access some Function which has not
    // been visited yet in the pass.
    for (Function &F : M) {
      const auto fnName = F.getName().str();
      fnCache[fnName] = &F;
    }
    for (auto &fn : fnCache) {
      utils::log(errs(), 0, 1, "Cached fn name", fn.first);
    }

    utils::log(errs(), 0, 2, "\n\nStartig the Function visit for the Module",
               M.getName());
    // Iterate over all functions and visit them
    for (Function &F : M) {
      // Visit only functions of interest
      const auto fnName = F.getName().str();
      if (fnName.find(targetOmpOutlinedFnNamePrefix) == std::string::npos) {
        utils::log(errs(), 0, 1, "Skipping the visit of function called",
                   fnName, "as of not interest");
        continue;
      }
      visitor(F, nullptr);
    }
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
PassPluginLibraryInfo getOmpDependencyFinderPlugInInfo() {
  return {LLVM_PLUGIN_API_VERSION, "OmpDependencyFinder", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "module-pass-and-analysis") {
                    MPM.addPass(OmpDependencyFinder());
                    return true;
                  }
                  return false;
                });
          }};
}

// This is the core interface for pass plugins. It guarantees that 'opt'
// will be able to recognize OmpDependencyFinder when added to the pass
// pipeline on the command line, i.e. via '-passes=fn-pass-and-analysis'
extern "C" LLVM_ATTRIBUTE_WEAK ::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getOmpDependencyFinderPlugInInfo();
}
