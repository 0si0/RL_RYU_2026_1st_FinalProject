# HandOccNet Notes for RL Final Project

Source: `/Users/ryu/Documents/논문 + 학습자료/HandOccNet- Occlusion-Robust 3D Hand Mesh Estimation Network.pdf`

## Core idea

HandOccNet addresses 3D hand mesh estimation when hands are severely occluded by objects. The paper's main insight is that occluded regions should not simply be discarded. Even low-confidence or partially occluded regions can contain useful secondary information if they are interpreted through their correlation with visible hand regions.

The paper uses:

- primary features: high-attention hand regions
- secondary features: low-attention or occluded regions
- FIT: injects primary hand information into correlated secondary/occluded regions
- SET: refines the injected feature map with self-attention
- MANO regression: predicts MANO pose, shape, joints, and mesh

For our RL assignment, we should not implement HandOccNet. The useful lesson is how to reason about incomplete or unreliable hand-object interaction signals.

## Assignment-level takeaways

The final project uses HO-Cap trajectories where the human hand manipulates an object. The assignment slides warn that the provided data is kinematically reconstructed and may include penetration, floating hands, or physically implausible contacts. HandOccNet adds another relevant point: in hand-object interaction, the object often hides important hand parts, so the hand target itself may be incomplete, ambiguous, or unreliable around contact.

This supports a robust policy design:

- Do not blindly trust every hand keypoint equally at every frame.
- Treat visible/structurally reliable hand parts as anchors.
- Use object motion and contact behavior to infer useful hand behavior when direct keypoint matching is ambiguous.
- Log and analyze failure cases involving occlusion/contact separately from generic tracking error.

## How this maps to our Isaac Lab project

In the paper:

- object occlusion makes direct hand reconstruction difficult
- visible hand regions help reconstruct occluded hand regions
- hand-object interaction datasets are the main benchmark

In our project:

- the object may physically block or perturb the robot hand during manipulation
- MANO keypoints are reference targets, but they may not correspond to a physically reachable robot hand configuration
- object motion is a critical additional signal that can tell whether the hand behavior is useful, even when exact hand keypoint tracking is imperfect

Practical interpretation:

- Use the MANO reference as a guide, not an absolute command.
- Use object trajectory and fingertip-object interaction as complementary signals.
- For hard optional sequences, prioritize robust contact and object motion instead of perfect keypoint imitation.

## Observation design ideas

The paper separates primary and secondary information. In our policy observation, a similar idea can be implemented without image features:

Primary/reliable state:

- robot hand joint positions and velocities
- current robot fingertip positions
- object position, rotation, linear velocity, angular velocity
- reference object position and rotation
- relative object error

Secondary/contextual state:

- all MANO keypoint targets
- robot-to-reference keypoint errors
- fingertip-to-object vectors
- palm-to-object vector
- contact force history
- frame/time progress

Useful observation additions after the baseline:

- fingertip positions relative to the object, not only world coordinates
- palm/root pose relative to the object
- previous action or action delta, if needed for smoothness
- contact force buffer or clipped contact indicators
- per-finger distance to object

The HandOccNet analogy: when direct hand target tracking is unreliable, let object-relative/contact observations provide context.

## Reward design ideas

HandOccNet is not a reward paper, but it motivates robust reward design under object occlusion/contact.

Baseline reward terms:

- hand keypoint imitation
- fingertip imitation
- object position tracking
- object rotation tracking
- object velocity tracking, optional and low weight
- action penalty

Robustness additions inspired by the paper:

1. Reliability-aware hand reward
   - Do not rely only on all 21 keypoints equally.
   - Add stable anchor rewards for wrist/MCP-like points.
   - Add separate fingertip reward for manipulation.

2. Contact-aware shaping
   - Add fingertip-object distance reward before contact.
   - Add clipped contact force reward when object should move.
   - Avoid unbounded force rewards, since the hand may learn to hit or crush the object.

3. Object-as-context reward
   - If the object follows the target trajectory, tolerate small local hand mismatch.
   - If hand keypoints look good but object does not move, increase object/contact terms.

4. Smooth recovery from bad states
   - Severe occlusion in the paper corresponds to ambiguous/incomplete state.
   - In RL, bad contact or drift can create similarly ambiguous recovery states.
   - Add early termination for large object/hand drift, but avoid terminating too aggressively before the policy can learn recovery.

## Optional sequence strategy

This paper is especially relevant for optional sequences because they are likely harder hand-object interactions.

Recommended optional workflow:

1. Train all sequences with the same baseline reward.
2. Inspect videos, not only scalar reward.
3. Categorize failure:
   - hand follows reference but object slips
   - object moves but fingers look implausible
   - contact never happens
   - contact happens but force is unstable
   - object is occluding/interrupting keypoint-like motion
4. Tune sequence-specific weights only after identifying the failure category.

Likely optional fixes:

- If contact never happens: increase fingertip-object distance reward.
- If contact is unstable: add action smoothness and contact force clipping.
- If object tracking fails: increase object position/rotation reward after contact is established.
- If hand shape collapses: add wrist/MCP anchor reward.
- If policy overfits early frames: randomize start frames or use frame-progress observation if implementation time allows.

## Experiment and report ideas

HandOccNet gives useful language for explaining why hand-object manipulation is hard:

- Hands are often occluded by objects during manipulation.
- Occluded or ambiguous hand parts should not be ignored.
- Object interaction provides context for reconstructing/planning plausible hand motion.
- Robustness should be evaluated on hand-object interaction datasets/sequences, not only clean isolated hand poses.

Possible report framing:

1. Problem difficulty
   - Provided HO-Cap data is kinematic.
   - Hand-object contact makes exact keypoint imitation insufficient.
   - Occlusion/contact can make some hand reference details unreliable.

2. Method
   - Use separate tracking terms for hand, fingertips, and object.
   - Use object-relative observations to provide interaction context.
   - Use contact/proximity rewards to bridge hand imitation and object manipulation.

3. Analysis
   - Compare hand tracking error and object tracking error.
   - Discuss whether contact improved object motion.
   - Use videos to show physically plausible manipulation, not only reward curves.

Useful report phrases:

- "occlusion-robust hand-object reasoning"
- "primary structural anchors and secondary contextual signals"
- "object-relative hand features"
- "interaction-aware reward shaping"
- "robustness under hand-object occlusion/contact"

## Diagnostics to log

Use TensorBoard metrics that reveal whether the policy is using hand-object context:

- `error/hand_kpt`
- `error/hand_anchor`
- `error/fingertip`
- `error/object_pos`
- `error/object_rot`
- `reward/hand`
- `reward/fingertip`
- `reward/object_pos`
- `reward/object_rot`
- `reward/contact`
- `metric/fingertip_object_dist`
- `metric/contact_force`
- `penalty/action`
- `metric/early_terminate_rate`

For videos/checkpoints, record which failure mode was observed for each sequence.

## Implementation details worth remembering from the paper

- HandOccNet predicts MANO pose parameters `theta` with dimension 48 and shape parameters `beta` with dimension 10.
- It supervises 2D heatmaps, MANO pose, MANO shape, 3D joints, and mesh vertices with L2 losses.
- It evaluates on hand-object interaction datasets such as HO-3D, FPHA, and Dex-YCB.
- The paper reports both Procrustes-aligned and non-Procrustes-aligned metrics, because global rotation/placement errors matter.
- FIT uses correlation between occluded/secondary regions and visible/primary hand regions.
- It avoids residual connections from the secondary query during injection because the goal is to replace unreliable secondary information with correlated primary information.

For our assignment, the important translation is:

- Separate reliable anchor signals from unreliable/full-detail signals.
- Do not let noisy/physically implausible reference details dominate policy learning.
- Preserve global/object-relative correctness as a first-class objective.

## What not to over-apply

- Do not implement FIT/SET or any image-based Transformer module.
- Do not add image occlusion processing; our environment already provides trajectory tensors.
- Do not spend time on MANO mesh regression unless the RL baseline and optional sequences are already working.
- Do not use Procrustes-style aligned errors as the only analysis, because the assignment cares about actual object position and rotation in the scene.
