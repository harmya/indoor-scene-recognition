# Results and Benchmarks
**Init Hyperparameters:**
- 10 epochs
- 32 Batch Size
- Adam optimizer
- 1e-5 Learning Rate
- ResNet18
#### Without Bounding Boxes: 58-60%ish on Val and Test
#### With Bounding Boxes: 62-64%ish on Val and Test
##### - With 40 epochs, with bounding boxes goes up to 73% on Test
##### - With 2e-5 lr, with bounding boxes plateaus around 74% on Test
##### - With 4e-5 lr, AdamW, with bounding boxes plateaus around 75% on Test