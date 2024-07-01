# Journey to write FSDP from scratch under 1k LOC.

Yes, ZeRO algorithm (used for torch FSDP and DeepSpeed) is complex. But at this point I've made minimal implementation of everything, that only logical next step is to implement ZeRO completely from scratch, with goal of reaching 0.4 MFU on h100s.

# The `/journey` Directory

Contains all the artifacts of the journey. Think of it more as a diary, kinda like chain of thought (hehe). `/journey` will get progressively larger and larger, all single-file implementations, which might eventually be the thing we want.