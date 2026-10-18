# Assignment 2 Report

Name:

UCID:

Tip: To include figures in your report, put them in the `assets` folder and
link to the relative path of the images. For a Matplotlib figure, call, for
example, `plt.savefig("assets/baseline_loss.png")` before
`plt.show()`.

## AI use declaration (1 mark)

List any sources you referred to or generative-AI assistance you used, and explain how you verified the contribution. If you did not use generative AI, write: `I did not use generative AI for this assignment.`

| Source | Source contribution | How I verified it |
| ------ | ------------------- | ----------------- |
|        |                     |                   |

## Questions

### 1. Baseline model (1 mark)

Report the validation results for the baseline model.

| Validation result   | Value |
| ------------------- | ----: |
| Objectness loss     |       |
| Classification loss |       |
| Bounding-box loss   |       |
| Total loss          |       |
| Precision           |       |
| Recall              |       |

Include the baseline training-loss plot here and briefly describe the training behaviour.

### 2. Training hyperparameter (2 marks)

State the one training hyperparameter you investigated. Compare the baseline setting with two alternatives while keeping other settings fixed where possible.

Chosen hyperparameter:

| Training setting | Validation loss | Precision | Recall |
| ---------------- | --------------: | --------: | -----: |
| Baseline:        |                 |           |        |
| Alternative 1:   |                 |           |        |
| Alternative 2:   |                 |           |        |

Plot the three training-loss curves together. Compare their shapes and validation results, explain plausible reasons for the differences, and identify which setting you would choose.

### 3. Architecture change (2 marks)

State the one aspect of the model architecture you changed and explain why you chose it. Keep the data split and training settings the same as the baseline.

Architectural change:

| Model                 | Validation loss | Precision | Recall |
| --------------------- | --------------: | --------: | -----: |
| Baseline              |                 |           |        |
| Modified architecture |                 |           |        |

Include the baseline and modified-model training-loss curves. Compare the validation results and loss curves, and explain why the change may have helped or hurt performance.

### 4. Model selection and final test result (1 mark)

Identify the final model and justify your choice using validation loss,
precision, and recall only. Discuss any tradeoff between precision and recall.
The test set must not be used for model selection.

Selected model:

Reason for selection:

| Final test result | Value |
| ----------------- | ----: |
| Total loss        |       |
| Precision         |       |
| Recall            |       |

### 5. Qualitative analysis (2 marks)

Include several test-set prediction visualizations here. The examples do not need to be failures.

Describe what the detector does well and analyze errors or limitations that you observe. For example, consider missed objects, false detections, incorrect classes, poorly localized boxes, overlapping objects, or small objects. Propose at least one reasonable way to improve the detector.

## Reflection (1 mark)

Include a sentence or two about:

- what you liked or disliked about the course so far, and
- what you found interesting, confusing, challenging, or motivating while working on this assignment.
