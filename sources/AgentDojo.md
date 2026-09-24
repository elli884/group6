# AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents

Link to paper: https://arxiv.org/pdf/2406.13352


AgentDojo,
an evaluation framework for agents that execute tools over untrusted data

link: https://agentdojo.spylab.ai/


AI agents could be used for various roles, such as digital assistants with access to emails and
calendars, or smart “operating systems” with access to coding environments and scripts 

## Agents and Prompt Injection Defenses

AgentDojo is designed as a benchmark environment to evaluate new agent designs that can defend
against prompt injection attacks. We thus provide a simple interface for implementing new agents.

## Resumee
Overall, the most capable model in a benign setting is Claude 3.5 Sonnet, closely followed by GPT-4o.
The former also provides a better tradeoff between utility and security against targeted attacks. For
the remaining experiments in this paper, we focus on GPT-4o as Claude 3.5 Sonnet was released after
the first version of this paper.

Our results indicate that AgentDojo poses challenges for both attackers and
defenders, and can serve as a live benchmark environment for measuring their respective progress.