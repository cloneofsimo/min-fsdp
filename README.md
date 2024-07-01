# Journey to write FSDP from scratch under 1k LOC.

Yes, ZeRO algorithm (used for torch FSDP and DeepSpeed) is complex. But at this point I've made minimal implementation of everything, that only logical next step is to implement ZeRO completely from scratch, with goal of reaching 0.4 MFU on h100s.

