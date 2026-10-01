// Wrapper only. The included upstream test file is byte-for-byte unmodified.
// Execute just testLangevin(), with fixed CPU thread count and force ordering.
#include "openmm/Platform.h"
#include <cstdlib>
#include <iostream>
using namespace OpenMM;
Platform& selectPlatform() {
    Platform::loadPluginsFromDirectory(std::getenv("OPENMM_PLUGIN_DIR"));
    const char* name=std::getenv("TEST_PLATFORM");
    Platform& result=Platform::getPlatformByName(name ? name : "CPU");
    if(result.getName()=="CPU") {
        result.setPropertyDefaultValue("Threads","1");
        result.setPropertyDefaultValue("DeterministicForces","true");
    }
    return result;
}
Platform& platform=selectPlatform();
void initializeTests(int,char**) {}
void runPlatformTests() {}
#define main upstream_unused_main
#include "upstream-TestCheckpoints.h"
#undef main
int main() {
    try {
        testLangevin();
        std::cout << "upstream testLangevin PASS on " << platform.getName() << std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cout << "upstream testLangevin FAIL on " << platform.getName() << ": " << e.what() << std::endl;
        return 1;
    }
}
