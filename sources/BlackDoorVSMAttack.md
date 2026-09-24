
# Comparison BlackDoorLLM vs. M-Attack-V2

| Property | BackdoorLLM | M-Attack-V2 |
|---|---|---|
| Main target | LLM behavior | Vision-language perception |
| Input | Primarily text | Image |
| Attack timing | Often training/model/inference manipulation | Input-time image manipulation |
| Requires model modification? | Often yes, depending on attack | No for target model |
| Main mechanism | Hidden conditional behavior | Adversarial pixel perturbation |
| Main threat | Persistent malicious behavior | Misperception of visual input |
| Black-box target | Not the central setting | Central to the attack |
| Supply-chain risk | Very high | Lower for the target model itself |
| Multimodal? | Primarily LLMs | LVLMs |

The important conceptual distinction is:

```text
BackdoorLLM:
"Make the model behave differently when triggered."


M-Attack-V2:
"Make the model perceive the input differently."
```

Both demonstrate that **LLM/AI security is broader than prompt injection**.

---
