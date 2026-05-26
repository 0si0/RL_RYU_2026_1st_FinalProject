# Hand4Whole++ / CHAM Notes for RL Final Project

Source: `/Users/ryu/Documents/논문 + 학습자료/Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator.pdf`

## Core idea

The paper targets a related but different problem: improving 3D whole-body pose estimation by combining a whole-body estimator with a hand-only estimator. The useful lesson for this project is not the CHAM architecture itself, but the decomposition:

- Fine finger articulation should be supervised with hand-specific signals.
- Wrist/root placement must stay globally consistent with the larger kinematic structure.
- Naively copying detailed hand predictions without global consistency can produce implausible poses.
- Rigid alignment using the wrist and MCP joints is a stable way to connect detailed hand geometry to a broader body/hand frame.

## Assignment-level takeaways

This paper is useful beyond reward/observation design. It gives language and structure for the whole final project:

- Problem framing: detailed hand motion and global physical consistency can conflict.
- Method framing: split the problem into local hand articulation, global hand/root consistency, and object/task consistency.
- Experiment framing: report hand accuracy and global/task accuracy separately, then explain tradeoffs.
- Failure analysis: distinguish "hand looks close to reference" from "hand-object motion is physically plausible."
- Optional sequence strategy: harder sequences may need stronger contact/global consistency terms instead of only more PPO training.

## How this maps to our Isaac Lab project

The paper's whole-body estimator is analogous to our global physics environment, while its hand-only estimator is analogous to the provided MANO hand trajectory.

- MANO trajectory gives detailed hand targets, but the assignment PPT warns that these trajectories are kinematically reconstructed and can be physically implausible.
- Isaac Lab provides the global physical consistency that the raw MANO trajectory lacks.
- Therefore, the policy should not blindly copy the hand trajectory. It should balance:
  - hand keypoint imitation
  - object trajectory tracking
  - fingertip/object interaction
  - palm/root stability
  - action smoothness
  - early termination for impossible drift

This is a strong report argument: "We treated the provided demonstration as a tracking guide, not as a fully physical ground truth."

## Code architecture implications

The paper uses a modular design: frozen pre-trained modules plus a small trainable modulator. For our project, the parallel idea is to keep code modular even though PPO is already provided.

Recommended project organization:

- Keep physics/action application code stable unless necessary.
- Put observation construction in one clearly inspectable block.
- Put intermediate errors in `_compute_intermediate_values`.
- Put reward composition in `compute_rewards`.
- Put reward scales and weights in `gr_env_cfg.py`, not scattered as unexplained constants.
- Split sequence configs for main/optional sequences instead of editing comments manually.
- Log every major reward/error term separately so the report can explain training behavior.

This helps both code quality and report clarity.

## Observation design ideas

The paper's central lesson is that local hand information needs global context. For our policy observation, useful categories are:

1. Current robot state
   - hand root/palm position and rotation
   - hand root linear/angular velocity
   - joint positions and velocities
   - fingertip positions and velocities

2. Current object state
   - object position and rotation
   - object linear/angular velocity

3. Reference target state
   - current or next MANO keypoints
   - current or next object position/rotation
   - optional object velocity reference

4. Relative features
   - robot keypoints minus reference keypoints
   - object pose error
   - fingertip positions relative to object
   - palm position relative to object
   - time/frame progress

Important: relative features are likely more useful than only absolute world coordinates because they directly tell the policy how to correct errors.

## Reward design ideas

1. Separate local hand detail from global object/task tracking.
   - Use all MANO/robot keypoints for general hand imitation.
   - Give fingertips extra weight because they drive contact and object motion.
   - Do not let fingertip tracking alone dominate, since that can produce unnatural palm/root motion.

2. Add wrist/root consistency terms.
   - The paper emphasizes that wrist orientation must be coherent with the full kinematic chain.
   - In our environment, this suggests monitoring palm/root pose and velocity smoothness, not only 21 keypoints.
   - A useful baseline reward can include palm-to-object or palm-to-reference consistency if keypoint imitation alone becomes unstable.

3. Use MCP/keypoint subsets as stable anchors.
   - The paper aligns hand meshes using the wrist and four MCP joints.
   - For this project, MCP-like MANO indices can be useful for a lower-noise "hand structure" reward in addition to fingertip reward.
   - Candidate MANO anchor indices from the assignment keypoints: wrist plus finger-base joints, likely `[0, 5, 9, 13, 17]` depending on the provided convention.

4. Avoid direct imitation that ignores physics.
   - The paper notes that high-quality hand-only predictions can be globally inconsistent.
   - In our RL setup, pure MANO keypoint tracking may make the robot chase non-physical reconstructed hand motion.
   - Therefore object tracking, contact/fingertip proximity, action smoothness, and early termination should balance raw keypoint imitation.

5. Use evaluation-style errors as TensorBoard logs.
   - The paper reports hand errors separately from full-body/global errors.
   - For our project, log separate terms:
     - `error/hand_kpt`
     - `error/fingertip`
     - `error/object_pos`
     - `error/object_rot`
     - `reward/hand`
     - `reward/fingertip`
     - `reward/object_pos`
     - `reward/object_rot`
     - `penalty/action`

6. Consider a staged reward schedule if training is unstable.
   - Early training: stronger hand/fingertip proximity shaping to help the hand approach useful contact.
   - Later training: stronger object position/rotation tracking.
   - This is similar in spirit to the paper's separation of wrist/global coherence from finger detail.

## Optional sequence strategy

The optional sequences are harder because they are longer and may include more complex contact or object motion. The paper's warning about hand-only predictions failing under interaction is especially relevant here.

Recommended strategy:

1. First run the same baseline reward on all three sequences.
2. Compare TensorBoard logs:
   - If hand reward improves but object reward stays low, increase contact/fingertip-object and object tracking terms.
   - If object reward improves but hand looks unnatural, increase hand/MCP-anchor/root consistency terms.
   - If reward is noisy and videos jitter, increase action smoothing penalty or reduce force/action gain.
3. Use sequence-specific config files for reward weights only after the shared baseline is understood.
4. Save separate best checkpoints and videos for each sequence.

Possible optional-specific additions:

- fingertip-object distance reward
- contact force reward, but clipped so the hand does not learn to smash the object
- palm-object distance regularizer
- MCP-anchor reward for stable hand shape
- stricter early termination when object falls or drifts far from reference

## Experiment and report ideas

The paper is useful for writing the report, not just coding.

Potential report structure:

1. Motivation
   - The given HO-Cap trajectory is kinematic and may contain penetration or floating hands.
   - A physical RL policy must balance imitation and object manipulation.

2. Observation design
   - Explain current robot state, object state, reference trajectory, and relative error features.
   - Emphasize that relative hand/object features provide global context.

3. Reward design
   - Separate hand imitation, fingertip/contact, object position, object rotation, velocity, and smoothness.
   - Explain why each term exists.

4. Training trend
   - Show total reward plus separate reward terms.
   - Discuss whether hand tracking and object tracking improved together or competed.

5. Qualitative results
   - Use play videos to analyze hand motion and object motion separately.
   - Mention failure modes: object slips, hand floats, hand tracks reference but contact is weak, or object follows while fingers look unnatural.

6. Optional sequence analysis
   - Compare sequence1/2/3.
   - Explain any sequence-specific tuning.

Useful paper-inspired wording:

- "local hand articulation"
- "global consistency"
- "physically plausible hand-object trajectory"
- "hand-specific detail alone is insufficient without task/global context"
- "wrist/palm/root coherence"
- "interaction-aware hand tracking"

## Evaluation and diagnostics

The paper separates hand errors from global/full-body errors. For this assignment, mirror that idea:

- Hand score proxy:
  - mean robot/MANO keypoint error
  - fingertip error
  - MCP-anchor error

- Object score proxy:
  - object position error
  - object rotation geodesic error
  - object velocity error, if useful

- Physical plausibility proxy:
  - fingertip-object distance
  - contact force magnitude
  - action magnitude
  - root/palm velocity
  - early termination rate

TensorBoard should show at least one metric from each group.

## Implementation details worth remembering from the paper

- Rigid alignment uses wrist and four MCP joints: index, middle, ring, pinky.
- MANO is more expressive for hand shape than SMPL-X in the paper's ablation.
- The paper avoids simply copying wrist orientation from the hand-only estimator because it lacks global context.
- They do not use Procrustes-aligned hand error for the main hand orientation analysis because it can hide global rotation/wrist errors.
- They use separate reference frames for losses depending on task/data type:
  - pelvis-relative for full body
  - wrist-relative for hand-only
  - right-wrist-relative for interacting hands

For our project, this supports using both:

- reference-relative/local errors for hand shape imitation
- world/object-relative errors for physical manipulation

## Practical coding implications

- Start with `gr_env.py` reward terms:
  - all-keypoint hand imitation
  - fingertip imitation
  - object position tracking
  - object rotation tracking
  - optional object velocity tracking
  - action smoothness penalty

- Add config weights in `gr_env_cfg.py` instead of hardcoding all scales.

- If optional sequences are unstable, try sequence-specific reward weights before changing PPO:
  - increase fingertip/contact reward when object does not move
  - increase object pose reward when hand moves well but object fails
  - increase action penalty or palm/root smoothness when hand motion jitters

- Possible extra reward after baseline:
  - MCP-anchor reward using wrist and MCP-like MANO keypoints
  - palm/object distance shaping
  - fingertip-object proximity or contact-force reward
  - early termination when object or hand drifts far from the reference

## What not to over-apply

Some parts of the paper are intellectually relevant but too large for this assignment:

- Do not implement CHAM, ViT feature modulation, or hand image cropping.
- Do not try to fit MANO/SMPL-X meshes inside Isaac Lab unless the baseline is already strong.
- Do not replace the assignment's provided PPO pipeline with a pose-estimation pipeline.
- Do not spend time on mesh boundary smoothing; our target is policy behavior and videos/checkpoints.

## Not directly useful for this project

- CHAM feature modulation and ViT token fusion are not needed for the RL assignment.
- SMPL-X/MANO mesh transfer is conceptually useful, but implementing mesh-level transfer is too much for the current Isaac Lab task.
- The paper is image-based pose estimation, while this project already provides trajectory tensors and uses physics-based RL.
