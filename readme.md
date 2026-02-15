## sref的metric
1.风格相似度：
/data/benchmark_metrics/style_similarity_batches.sh。两个指标都可以改造这个脚本
1.1 oneig
1.2 csd

VLM打分，分级从0-5，需要有reasoning
/data/benchmark_metrics/triplet_similarity/style_similarity.sh

2.内容相似度：
有两个指标
2.1 /data/benchmark_metrics/benchmark_metrics/cas_sim.py 
2.2 /data/benchmark_metrics/content_similarity_batches.sh backend选择dinov2

VLM打分，分级从0-5.需要有reasoning
/data/benchmark_metrics/triplet_similarity/content_similarity.sh

## Sref + Cref 
这里是拒绝采样，采样三次，然后要求两次的执行度要大于0.5，而且要有两次为true，输出的指标是判别为true的置信度的均值。
可能需要改一下
1.1风格相似度： qwen标注
/data/benchmark_metrics/sref_pipeline/triplet_qwen_style_judge.sh 
1.2内容相似度： qwen标注
/data/benchmark_metrics/sref_pipeline/triplet_qwen_content_judge.sh


3.指标遵循的metric
这个需要诺哥实现一下
3.1 clip-i, clip-t
3.2 vlm的判别，可以是qwen或者gpt。参考这个链接里的prompt
https://github.com/VectorSpaceLab/OmniGen2/blob/main/omnicontext/prompt_generator.py
_prompts_0shot_in_context_generation_rule_PF_Scene = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, **regardless of whether subject identities or the scene are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1–3:** The image responds to the instruction mostly incorrectly.
* **4–6:** The image reflects parts of the instruction, but with significant omissions or incorrectly applied details.
* **7–9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.
* Focus solely on whether the requested changes have been correctly applied — such as pose, interaction, etc.
* Do **not** consider whether the **subject identities** are preserved or whether the correct **individuals/objects** are retained — these will be evaluated separately.
* Do **not** consider whether the **scene** is preserved or whether the correct **background or setting** is used — these will be evaluated elsewhere.
* Do **not** assess artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

Editing instruction: <instruction>
"""

4.美学评分
q-align-score
/data/benchmark_metrics/aesthetic/demo_aeshetic_scoreer.py