# Not what you’ve signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection

https://arxiv.org/pdf/2302.12173

Github repo:
https://github.com/greshake/llm-security

- With LLM-integrated applications, adversaries could control the LLM, without direct access, by indirectly injecting it with prompts placed within sources retrieved at inference time

- functionalities of recent LLMs can be flexible modulated via natural language prompts

## Prompt Injection (PI)

- enable attackers to override original instructions and employed controls

- pose a significant threat to the security of LLMs

## Indirect Prompt Injection

- remotely affect other users’ systems by strategically inject-
ing the prompts into data likely to be retrieved at inference time -> if retrieved and ingested, these prompts can indirectly control the
model

- can lead to full compromise of the model at inference time analogous to traditional
security principles

### Injection Method
- Passive methods (by retrieval)
- Active methods (e.g. mails)
- User-driven injections
- Hidden injections

### Affected parties
- end users
- developers
- automated systems
- the LLM itself (availability)

## LLM-Integrated Applications:

- can offer interactive chat
and summary of the retrieved search results or documents and perform actions on behalf of the user by calling other APIs

- paper argues that this AI-integration race is not
accompanied by adequate guardrails and safety evaluations

## Jailbraiking

- Robustness problem between instructions and model behavior

- OpenAI reports that jail-
breaking is harder for GPT-4 but still possible 


## Injection Methods

### Passive Methods

These methods rely on retrieval to deliver
injections. For example, for search engines, the prompts could be
placed within public sources (e.g., a website or social media posts)
that would get retrieved by a search query. Attackers could use Search Engine Optimization (SEO) techniques to promote their poi-
sonous websites. Moreover, Microsoft Edge has a Bing Chat sidebar;
if enabled by the user, the model can read the current page and,
e.g., summarize it. We found that any prompts/instructions writ-
ten on a page (while being invisible to the user) can be effectively
injected and affect the model. For code auto-completion models,
the prompts could be placed within imported code available via
code repositories. Even with offline models that retrieve personal
or documentation files (e.g., the ChatGPT Retrieval Plugin [9]), the
prompts could be injected by poisoning the input data.

### Hidden Injections
To make the injections more stealthy, at-
tackers could use multiple exploit stages, where an initial smaller
injection instructs the model to fetch a larger payload from an-
other source. Additionally, improvements in models’ capabilities
and supported modalities could open new doors for injections. For
example, with multi-modal models (e.g., GPT-4), the prompts could
be hidden in images (see Figure 28 for an example we constructed).
To circumvent filtering, prompts can also be encoded. Moreover,
instead of feeding prompts to the model directly, they could be the
result of Python programs that the model is instructed to run –
enabling encrypted payloads to pass safeguards. These possibilities
would make the prompts harder to detect.

## Key Message

- Models can currently act as a vulnerable,
easy-to-manipulate, intermediate layer between users and
information, which users might nevertheless overrely on. I.e.,
the model’s functionality itself can be attacked

- As LLMs themselves are in charge of when
and how to issue other API calls and process their outputs, the
input and output operations are vulnerable to manipulation
and sabotage.

## Resumee

a clear-cut defense against these problems is, at least, difficult to
achieve