#pragma once

// `moeshift experts extract|verify|strip`: prepares a GGUF's routed experts without Python.
// argv[0] is the command name; returns the process exit code.
int run_les_experts(int argc, char ** argv);
