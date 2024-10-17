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
#include <unordered_set>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/InlineCost.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instruction.h>
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
const std::string targetOmpOutlinedFnNamePreifx = ".omp_outlined";

constexpr uint8_t OPEN_FLAG = 0;
constexpr uint8_t CLOSE_FLAG_WITH_DEPS = 1;
constexpr uint8_t CLOSE_FLAG = 2;

void visitor(Function &F, std::vector<unsigned> const *entryPointInputs,
             bool clearDeps = true);

// key: fn ID
// value.first: in_deps
// value.second: out_deps
std::map<unsigned, std::pair<std::set<unsigned>, std::set<unsigned>>>
    dependentTasks;

std::unordered_map<std::string, Function *> fnCache;

std::unordered_map<unsigned, std::vector<unsigned>> taskEntryInputs;

std::unordered_set<unsigned> freeTasks;

const std::unordered_set<uint8_t> outDepsFlags = {2, 3};

const std::unordered_set<uint8_t> inDepsFlags = {1};

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

std::string parseUnamedNameFromInstruction(std::string const &str) {
  return retrieveLabelFromChar(str, '%');
};

std::string parsePtrToIntSource(std::string const &str) {
  unsigned i = 0;
  while (i < str.size() && str[i] != '%') {
    i++;
  }
  return retrieveLabelFromChar(str.substr(i + 1), '%');
};

namespace utils {
template <typename... T>
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

/// Retrive the instructions wrapping a task creation and submission
/// A single basic block may contain multiple task creation instructions
/// @param block A basic block
/// @param includeAllBeforeTaskAllocation A flag to control if to include all
/// instructions or not
/// @return A list of possbile instructions containing task creation
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
/// StoreType::PointerStore for dependencies souces and @ref
/// StoreType::ValueStore for dependencies types
/// @param GEP A pointer to a get element pointer instance
/// @return The label stored in the GEP reference
template <StoreType T> int GEPStoreUses(const GetElementPtrInst *GEP) {
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
      return std::stoi(label);
    }
  }
  return -1;
}

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

/// Find the inputs label to a omp task entry lambda
/// @param I The omp task alloc instruction
/// @return A @ref std::vector representing the inputs labels
std::vector<unsigned> findTaskEntrypointInputs(
    Instruction &I, std::unordered_map<unsigned, unsigned> const &labelMap) {
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
                    &labelMap](auto *storeInst, const unsigned index) {
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
    inputs.push_back(labelMap.find(numericalValue) == labelMap.end()
                         ? numericalValue
                         : labelMap.at(numericalValue));
  };

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

void ompDependencyFinderChilds(std::vector<unsigned> const &inputs,
                               std::string const &entryPointName) {
  // Recover the cached function
  Function *fn = fnCache[entryPointName];
  if (fn) {
    utils::log(errs(), 2, 1, "Recovered child fn", fn->getName());
  } else {
    utils::log(errs(), 2, 1, "Entry point", entryPointName,
               "not found in cache");
  }

  // Log
  utils::log(errs(), 2, 1, "Inputs to the entry point:");
  utils::log(errs(), 2, 0, "[");
  for (auto input : inputs) {
    const std::string formattedInput = "%" + std::to_string(input);
    utils::log(errs(), 2, 0, formattedInput);
  }
  utils::log(errs(), 2, 1, "]");

  visitor(*fn, &inputs, false);
}

/// Find omp dependencies from a list of instructions containing a sequnce
/// of omp task calls. Dependencies are discovered by leveraging GEP calls
/// which populates omp deps structs. Other instructions might be present
/// inside the input sequence and are filtered out. This pass will also save
/// for each omp task entry lambda the relative inputs.
/// @param instructions A @ref std::vector of @ref Instruction
/// @param targetStructName The struct name to search
/// @param entryPointInputs A vector of the current task entry lambda inputs
/// labels
template <bool ShowInbounds = false>
void ompDependenciesFinder(std::vector<Instruction *> const &instrunctions,
                           std::vector<Instruction *> const &allFnInstructions,
                           const std::string &targetStructName,
                           std::vector<unsigned> const *entryPointInputs) {
  std::unordered_map<unsigned, unsigned> inputMap;

  int foundDepsLabel = -1;
  std::string ompTaskEntryLamdaName = "";
  std::string stringBuff = "";
  raw_string_ostream stream(stringBuff);

  utils::log(errs(), 2, 1, "\nFirst instruction:", *(instrunctions[0]));
  utils::log(errs(), 2, 1,
             "Last instruction:", *(instrunctions[instrunctions.size() - 1]));

  for (auto &inst : instrunctions) {
    if (CallInst *ompTaskAPICall = dyn_cast<CallInst>(inst)) {
      auto calledFnName = ompTaskAPICall->getCalledFunction()->getName().str();
      // It's not either a task creation open call or close call
      if ((calledFnName != targetOmpTaskAllocName) &&
          (calledFnName != targetOmpTaskSubmissionDepsName) &&
          (calledFnName != targetOmpTaskSubmissionName)) {
        continue;
      } else if (calledFnName == targetOmpTaskAllocName) {
        // Extract the executed lambda reference
        const auto noOperands = ompTaskAPICall->getNumOperands();
        // The task entry lambda argmument is stored in the -2 index
        Value *ompTaskEntryLambdaArg =
            ompTaskAPICall->getArgOperand(noOperands - 2);
        utils::log(errs(), 2, 1,
                   "Lambda called name:", ompTaskEntryLambdaArg->getName());
        ompTaskEntryLamdaName = ompTaskEntryLambdaArg->getName().str();

        // Find the task inputs. Dependencies data will figure in this
        // input vector
        taskEntryInputs[std::stoi(getOmpTaskEntryId(ompTaskEntryLamdaName))] =
            findTaskEntrypointInputs(*inst, inputMap);

        continue;
      } else if (calledFnName == targetOmpTaskSubmissionName) {
        // When encountering a closing API call, if it's a task with no
        // deps save it
        freeTasks.insert(std::stoi(getOmpTaskEntryId(ompTaskEntryLamdaName)));
        // The inputs will have the orignal labels defined inside the parallel
        // region entry point (omp_outlined) to have a consistent global
        // identification
        ompDependencyFinderChilds(taskEntryInputs[std::stoi(getOmpTaskEntryId(
                                      ompTaskEntryLamdaName))],
                                  ompTaskEntryLamdaName);
      } else if (calledFnName == targetOmpTaskSubmissionDepsName) {
        // The inputs will have the orignal labels defined inside the parallel
        // region entry point (omp_outlined) to have a consistent global
        // identification
        ompDependencyFinderChilds(taskEntryInputs[std::stoi(getOmpTaskEntryId(
                                      ompTaskEntryLamdaName))],
                                  ompTaskEntryLamdaName);
      }
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(inst)) {
      GEP->print(stream);
      utils::log(errs(), 2, 1,
                 "Found a GetElementPtr instruction: ", stringBuff);
      stringBuff.clear();

      // Optionally, check if it's 'inbounds'
      if constexpr (ShowInbounds) {
        if (GEP->isInBounds()) {
          utils::log(errs(), 2, 1, "GEP instruction is 'inbounds'");
        }
      }

      // Check if the GEP does access the struct of interest
      if (GEPIsArrayOfSpecificStruct(GEP, targetStructName)) {
        // kaixi: Do refer to technical docs for these accesses
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
            int label = GEPStoreUses<StoreType::PointerStore>(GEP);
            // The label being stored in the GEP reference will be the output
            // alias from a `ptrtoint` instruction. The real data source is
            // available inside the `ptrtoint` call which is before this point.
            // Recover it.
            label = std::stoi(recoverPtrToIntSource(allFnInstructions,
                                                    std::to_string(label)));
            foundDepsLabel = label;
            // Map the label to a parent source if available
            if (inputMap.find(foundDepsLabel) != inputMap.end()) {
              foundDepsLabel = inputMap.at(foundDepsLabel);
            }
          } else if (value == 2) {
            utils::log(errs(), 2, 1,
                       "GEP instruction on the dependency type field");
            const int label = GEPStoreUses<StoreType::ValueStore>(GEP);

#ifdef ENABLE_ASSERTIONS
            assert(foundDepsLabel != -1);
#endif
            // We save our dependency immediately here at the end of the
            // target GEP and not waiting for the task submission as there
            // might be more than one dependencies creation between the
            // task alloc call and task sumbission call.
            utils::log(errs(), 2, 1,
                       "Found a dependency. {Source:", foundDepsLabel,
                       ", Type:", label, "}");
            if (inDepsFlags.find(label) != inDepsFlags.end()) {
              utils::log(errs(), 2, 1, "Assigning dependentTasks. Holder:",
                         ompTaskEntryLamdaName,
                         " input dep with:", foundDepsLabel);
              dependentTasks[std::stoi(
                                 getOmpTaskEntryId(ompTaskEntryLamdaName))]
                  .first.insert(foundDepsLabel);
            } else if (outDepsFlags.find(label) != outDepsFlags.end()) {
              utils::log(errs(), 2, 1, "Assigning dependentTasks. Holder:",
                         ompTaskEntryLamdaName,
                         " output dep with:", foundDepsLabel);
              dependentTasks[std::stoi(
                                 getOmpTaskEntryId(ompTaskEntryLamdaName))]
                  .second.insert(foundDepsLabel);
            }
          }
        }
      } else if (GEPIsSpecificStruct(GEP, targetOmpAnonStructName)) {
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
  std::unordered_map<unsigned, std::vector<unsigned>> depsCount;
  for (auto &kv : dependentTasks) {
    const std::string formattedLambdaName = 'L' + std::to_string(kv.first);
    utils::log(errs(), 0, 1, "\n\nLambda name id:", formattedLambdaName);
    if (taskEntryInputs[kv.first].size()) {
      utils::log(errs(), 0, 1, "Inputs:");
      utils::log(errs(), 0, 0, "[");
      for (auto &input : taskEntryInputs[kv.first]) {
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
    std::set<unsigned> uniqueParents;
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
        const std::string formattedLambdaName = 'L' + std::to_string(parent);
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
    const std::string formattedLambdaName = 'L' + std::to_string(task);
    utils::log(errs(), 0, 0, formattedLambdaName);
  }
  utils::log(errs(), 0, 1, "]");
}

/// This method implements what the pass does
/// @param F The @ref llvm:Function to visit
/// @param entryPointInputs The current taks entry point lambda inputs arguments
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
    // at omp_outlined leve. In this case we don't want to include all the
    // instructions before that task alloc.  In case this visitor is called in a
    // nested call we want to include the informations before the task alloc as
    // they contains data for task entry inputs and dependencies sources.
    const auto parsedInstructions = taskWithDepsInstructions(BB, !clearDeps);
    utils::log(errs(), 1, 1, "Basic Block task with dependentTasks #:",
               parsedInstructions.size());

    std::vector<Instruction *> allFnInstructions;
    for (auto &i : parsedInstructions) {
      allFnInstructions.insert(allFnInstructions.end(), i.begin(), i.end());
    }

    for (auto &i : parsedInstructions) {
      // for (auto &j : i) {
      //   errs() << *j << "\n";
      // }
      ompDependenciesFinder(i, allFnInstructions, targetOmpDepsStructName,
                            entryPointInputs);
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
      if (fnName.find(targetOmpOutlinedFnNamePreifx) == std::string::npos) {
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
