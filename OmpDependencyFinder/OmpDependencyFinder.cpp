//=============================================================================
// FILE:
//    OmpDependencyFinder.cpp
//
// DESCRIPTION:
//    Visits all functions in a module, prints their names, the number of
//    arguments, function blocks and instructions via stderr. Strictly speaking,
//    this is an analysis pass (i.e. the functions are not modified). However,
//    in order to keep things simple there's no 'print' method here (every
//    analysis pass should implement it).
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
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

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
const std::string targetCloseOmpCall = "__kmpc_omp_task_with_deps";
const std::string targetOmpDepsStructName = "struct.kmp_depend_info";

constexpr uint8_t OPEN_FLAG = 0;
constexpr uint8_t CLOSE_FLAG = 1;

std::unordered_map<unsigned, std::vector<unsigned>> deps;

std::unordered_set<uint8_t> inDepsFlags = {1, 3};

std::unordered_set<uint8_t> outDepsFlags = {2};

bool isTarget(Instruction const &inst, const uint8_t flag) {
  StringRef instruction_name = inst.getName();
  StringRef opcode_name = inst.getOpcodeName();
  if (auto *CI = dyn_cast<CallInst>(&inst)) {
    StringRef name = CI->getCalledFunction()->getName();
    if (flag == OPEN_FLAG && name.data() == targetOpenOmpCall) {
      return true;
    } else if (flag == CLOSE_FLAG && name.data() == targetCloseOmpCall) {
      return true;
    }
  }
  return false;
}

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
    } else if (inside && isTarget(inst, CLOSE_FLAG)) {
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
    errs() << "  The type being indexed is an array.\n";
    // Get the element type of the array
    ArrayType *arrayType = cast<ArrayType>(sourceType);
    Type *elementType = arrayType->getElementType();

    // Check if the element type is a struct
    if (elementType->isStructTy()) {
      StructType *structType = cast<StructType>(elementType);

      // Compare the struct name with the one we are looking for
      if (structType->getName().data() == structName) {
        errs() << "  This GEP is accessing an array of the struct: "
               << structName << "\n";
        return true;
      } else {
        errs() << "  This GEP is accessing an array of the struct different "
                  "from the target struct: "
               << structType->getName() << "\n";
      }
    }
  } else {
    errs() << "  The type being indexed is not an array.\n";
  }
  return false;
}

void GEPUses(const GetElementPtrInst *GEP) {
  errs() << "  Analyzing uses of GEP instruction:\n";
  GEP->print(errs());
  errs() << "\n";

  // Iterate through all uses of the GEP instruction
  for (const Use &U : GEP->uses()) {
    const User *user =
        U.getUser(); // Get the user (instruction that uses the GEP result)

    // Check if the GEP result is used in a store instruction
    if (const StoreInst *storeInst = dyn_cast<StoreInst>(user)) {
      errs() << "  GEP result is being stored in memory:\n";
      storeInst->print(errs());
      errs() << "\n";
    }
    // Check if the GEP result is used in a load instruction
    else if (const LoadInst *loadInst = dyn_cast<LoadInst>(user)) {
      errs() << "  GEP result is being loaded from memory:\n";
      loadInst->print(errs());
      errs() << "\n";
    }
    // Check if the GEP result is passed as an argument to a function call
    else if (const CallInst *callInst = dyn_cast<CallInst>(user)) {
      errs()
          << "  GEP result is being used as an argument in a function call:\n";
      callInst->print(errs());
      errs() << "\n";
    }
    // Handle other types of instructions
    else {
      errs() << "  GEP result is being used in:\n";
      user->print(errs());
      errs() << "\n";
    }
  }
}

enum StoreType { PointerStore, ValueStore };

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

  errs() << "  Analyzing store uses of GEP instruction:\n";
  GEP->print(errs());
  errs() << "\n";

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
      errs() << "  Stored value" << *storedValue << "\n";
      std::string outputStr;
      raw_string_ostream stream(outputStr);
      storedValue->print(stream);

      std::string label;
      if constexpr (T == StoreType::PointerStore) {
        label = getPtrLabel(outputStr);
      } else {
        label = getValueLabel(outputStr);
      }
      return std::stoi(label);
    }
  }
  return -1;
}

// Find an omp dependency from a list of instructions containing a sequnce of
// GEP calls
void GEPDependencyFinder(std::vector<Instruction *> instrunctions) {
  int foundDepsLabel = -1;
  for (auto &inst : instrunctions) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(inst)) {
      errs() << "Found a GetElementPtr instruction:\n";
      GEP->print(errs());
      errs() << "\n";

      // Optionally, check if it's 'inbounds'
      if (GEP->isInBounds()) {
        errs() << "  This GEP instruction is 'inbounds'.\n";
      }

      if (GEPIsArrayOfSpecificStruct(GEP, targetOmpDepsStructName)) {
        // Find uses of GEP if the last operand is zero (the last index accesses
        // the struct fields, and we need to access the first field)
        const auto lastIndex = GEP->getNumOperands() - 1;
        Value *value = GEP->getOperand(lastIndex);
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(value)) {
          const auto value = CI->getValue();
          if (value == 0) {
            errs() << "  GEP instruction on the dependency source field. Finding its "
                      "source...\n";
            const int label = GEPStoreUses<StoreType::PointerStore>(GEP);
            foundDepsLabel = label;
          } else if (value == 2) {
            errs() << "  GEP instruction on the dependency type field. Finding its "
                      "value...\n";
            const int label = GEPStoreUses<StoreType::ValueStore>(GEP);
            errs() << "  Found a dependency. {Source: " << foundDepsLabel
                    << ", Type: " << label << "}\n";
          }
        }
      }
    }
  }
}

// This method implements what the pass does
void visitor(Function &F) {
  errs() << "################ START OF FUNCTION ################\n\n";
  errs() << "(llvm-tutor) Hello from: " << F.getName() << "\n";
  errs() << "(llvm-tutor)   number of arguments: " << F.arg_size() << "\n";
  // Iterate over basic blocks in the function
  for (BasicBlock &BB : F) {
    errs() << "(llvm-tutor)    Basic Block: " << BB.getName() << "\n";
    const auto parsedInstructions = taskWithDepsInstructions(BB);
    errs() << "(llvm-tutor)    Basic Block task with deps: "
           << parsedInstructions.size() << "\n";

    for (auto &i : parsedInstructions) {
      for (auto &j : i) {
        errs() << *j << "\n";
      }
      errs() << "\n################ START GEP ANALYSIS ################\n";
      GEPDependencyFinder(i);
      errs() << "\n################ END GEP ANALYSIS ################\n";
    }
    errs() << "################ END OF BLOCK ################\n";
  }
  errs() << "\n################ END OF FUNCTION ################\n\n";
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
// be able to recognize OmpDependencyFinder when added to the pass pipeline on the
// command line, i.e. via '-passes=fn-pass-and-analysis'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getOmpDependencyFinderPlugInInfo();
}
