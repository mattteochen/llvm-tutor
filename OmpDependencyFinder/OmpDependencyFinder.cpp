//=============================================================================
// FILE:
//    OmpDependencyFinder.cpp
//
// DESCRIPTION:
//    Find the static computational graph of a OpenMP task program
//
// USAGE:
//    New PM
//      opt -load-pass-plugin=libHelloWorld.dylib -passes="fn-pass-and-analysis"
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

const std::string targetOpenOmpCall = "__kmpc_omp_task_alloc";
const std::string targetCloseDepsOmpCall = "__kmpc_omp_task_with_deps";
const std::string targetCloseOmpCall = "__kmpc_omp_task";
const std::string targetOmpDepsStructName = "struct.kmp_depend_info";

constexpr uint8_t OPEN_FLAG = 0;
constexpr uint8_t CLOSE_FLAG_WITH_DEPS = 1;
constexpr uint8_t CLOSE_FLAG = 2;

// key: fn ID
// value.first: in_deps
// value.second: out_deps
std::map<unsigned, std::pair<std::set<unsigned>, std::set<unsigned>>>
    dependentTasks;

std::unordered_set<unsigned> freeTasks;

const std::unordered_set<uint8_t> outDepsFlags = {2, 3};

const std::unordered_set<uint8_t> inDepsFlags = {1};

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
    if (flag == OPEN_FLAG && name.data() == targetOpenOmpCall) {
      return true;
    } else if (flag == CLOSE_FLAG_WITH_DEPS &&
               name.data() == targetCloseDepsOmpCall) {
      return true;
    } else if (flag == CLOSE_FLAG && name.data() == targetCloseOmpCall) {
      return true;
    }
  }
  return false;
}

/// Retrive the instructions wrapping a task creation and submission
/// A single basic block may contain multiple task creation instructions
/// @param block A basic block
/// @return A list of possbile instructions containing task creation
std::vector<std::vector<Instruction *>>
taskWithDepsInstructions(BasicBlock &block) {
  bool inside = 0;
  std::vector<Instruction *> instructions;
  std::vector<std::vector<Instruction *>> blockInstructions;
  for (Instruction &inst : block) {
    if (!inside && isTarget(inst, OPEN_FLAG)) {
      instructions.clear();
      inside = 1;
      instructions.push_back(&inst);
    } else if (inside && (isTarget(inst, CLOSE_FLAG_WITH_DEPS) ||
                          (isTarget(inst, CLOSE_FLAG)))) {
      inside = 0;
      instructions.push_back(&inst);
      blockInstructions.push_back(std::vector(instructions));
    } else if (inside) {
      instructions.push_back(&inst);
    }
  }
  return blockInstructions;
}

// Function to check if the GEP is accessing an array of a specific struct
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
/// @param GEP A pointer to a @ref llvm::GetElementPtrInst instance
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

/// Find an omp dependency from a list of instructions containing a sequnce of
/// GEP calls. Other instructions might be present inside the input sequence and
/// are filtered out.
/// @param instructions A @ref std::vector of @ref llvm::Instruction
/// @param targetStructName The struct name to search
template <bool ShowInbounds = false>
void GEPDependencyFinder(std::vector<Instruction *> instrunctions,
                         const std::string &targetStructName) {
  int foundDepsLabel = -1;
  std::string ompTaskEntryLamda = "";
  std::string stringBuff = "";
  raw_string_ostream stream(stringBuff);

  utils::log(errs(), 2, 1, "\nFirst instruction:", *(instrunctions[0]));
  utils::log(errs(), 2, 1,
             "Last instruction:", *(instrunctions[instrunctions.size() - 1]));

  for (auto &inst : instrunctions) {
    if (CallInst *ompTaskAPICall = dyn_cast<llvm::CallInst>(inst)) {
      auto calledFnName = ompTaskAPICall->getCalledFunction()->getName().str();
      // It's not either a task creation open call or close call
      if ((calledFnName != targetOpenOmpCall) &&
          (calledFnName != targetCloseDepsOmpCall) &&
          (calledFnName != targetCloseOmpCall)) {
        continue;
      } else if (calledFnName == targetOpenOmpCall) {
        // Extract the executed lamda reference
        const auto noOperands = ompTaskAPICall->getNumOperands();
        // The task entry lambda argmument is stored in the -2 index
        Value *ompTaskEntryLambdaArg =
            ompTaskAPICall->getArgOperand(noOperands - 2);
        utils::log(errs(), 2, 1,
                   "Lambda called name:", ompTaskEntryLambdaArg->getName());
        ompTaskEntryLamda = ompTaskEntryLambdaArg->getName().str();
        continue;
      } else if (calledFnName == targetCloseOmpCall) {
        // When encountering a closing API call, if it's a task with no deps
        // save it
        freeTasks.insert(std::stoi(getOmpTaskEntryId(ompTaskEntryLamda)));
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
            const int label = GEPStoreUses<StoreType::PointerStore>(GEP);
            foundDepsLabel = label;
          } else if (value == 2) {
            utils::log(errs(), 2, 1,
                       "GEP instruction on the dependency type field");
            const int label = GEPStoreUses<StoreType::ValueStore>(GEP);

#ifdef ENABLE_ASSERTIONS
            assert(foundDepsLabel != -1);
#endif
            // We save our dependency immediately here at the end of the target
            // GEP and not waiting for the task submission as there might be
            // more than one dependencies creation between the task alloc call
            // and task sumbission call.
            utils::log(errs(), 2, 1,
                       "Found a dependency. {Source:", foundDepsLabel,
                       ", Type:", label, "}");
            if (inDepsFlags.find(label) != inDepsFlags.end()) {
              utils::log(errs(), 2, 1,
                         "Assigning dependentTasks. Holder:", ompTaskEntryLamda,
                         " input dep with:", foundDepsLabel);
              dependentTasks[std::stoi(getOmpTaskEntryId(ompTaskEntryLamda))]
                  .first.insert(foundDepsLabel);
            } else if (outDepsFlags.find(label) != outDepsFlags.end()) {
              utils::log(errs(), 2, 1,
                         "Assigning dependentTasks. Holder:", ompTaskEntryLamda,
                         " output dep with:", foundDepsLabel);
              dependentTasks[std::stoi(getOmpTaskEntryId(ompTaskEntryLamda))]
                  .second.insert(foundDepsLabel);
            }
          }
        }
      }
    }
  }
}

/// This method implements what the pass does
/// @param F The @ref llvm:Function to visit
void visitor(Function &F) {
  // Clear our dependentTasks holder for now as it will be only printed to
  // standard error
  dependentTasks.clear();
  freeTasks.clear();
  utils::log(errs(), 0, 2,
             "################ START OF FUNCTION ################");
  utils::log(errs(), 0, 1, "Function name:", F.getName(),
             "#args:", F.arg_size());

  // Iterate over basic blocks in the function
  for (BasicBlock &BB : F) {
    utils::log(errs(), 0, 1,
               "\n################# START OF BLOCK ##################");
    // For each basic block retrieves the instructions sequence defining a task
    // creation and submission
    const auto parsedInstructions = taskWithDepsInstructions(BB);
    utils::log(errs(), 1, 1, "Basic Block task with dependentTasks #:",
               parsedInstructions.size());

    for (auto &i : parsedInstructions) {
      // for (auto &j : i) {
      //   errs() << *j << "\n";
      // }
      GEPDependencyFinder(i, targetOmpDepsStructName);
    }
    utils::log(errs(), 0, 1,
               "################## END OF BLOCK ###################");
  }

  // Print our the task graph (to not be confused with OpenMP task graph)
  utils::log(errs(), 1, 1, "\nDependency graph:");
  std::unordered_map<unsigned, std::vector<unsigned>> depsCount;
  for (auto &kv : dependentTasks) {
    utils::log(errs(), 0, 1, "\nLambda name id:", kv.first);
    utils::log(errs(), 0, 0, "In dependentTasks:");
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
      utils::log(
          errs(), 0, 1,
          "\nThis task depends on the termination of the following parents:");
      utils::log(errs(), 0, 0, "[");
      for (auto &parent : uniqueParents) {
        utils::log(errs(), 0, 0, parent);
      }
      utils::log(errs(), 0, 1, "]");
    } else {
      utils::log(errs(), 0, 1, " ");
    }

    utils::log(errs(), 0, 0, "Out dependentTasks:");
    for (auto &d : kv.second.second) {
      depsCount[d].push_back(kv.first);
      const std::string formatted = '%' + std::to_string(d);
      utils::log(errs(), 0, 0, formatted);
    }
  }

  utils::log(errs(), 0, 1, "\nFree tasks:");
  utils::log(errs(), 0, 0, "[");
  for (auto &task : freeTasks) {
    utils::log(errs(), 0, 0, task);
  }
  utils::log(errs(), 0, 1, "]");

  utils::log(errs(), 0, 2,
             "\n################ END OF FUNCTION ##################");
}

// New PM implementation
struct OmpDependencyFinder : PassInfoMixin<OmpDependencyFinder> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    visitor(F);
    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getOmpDependencyFinderPlugInInfo() {
  return {LLVM_PLUGIN_API_VERSION, "OmpDependencyFinder", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "fn-pass-and-analysis") {
                    FPM.addPass(OmpDependencyFinder());
                    return true;
                  }
                  return false;
                });
          }};
}

// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize OmpDependencyFinder when added to the pass pipeline on
// the command line, i.e. via '-passes=fn-pass-and-analysis'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getOmpDependencyFinderPlugInInfo();
}
