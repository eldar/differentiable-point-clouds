## Generating Ground Truth Point Clouds

The code for dense sampling of points from the ShapeNet meshes is from the repository of [Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction](https://github.com/chenhsuanlin/3D-point-cloud-generation).
Edit the script `data/generate_ground_truth.sh` to specify the path to ShapeNet V1 as well as the subset of the models (`val` or `test`).
Then execute the following commands:

```bash
cd data
./generate_ground_truth.sh 03001627
source densify_03001627_val.txt
```

This is a rather slow process, and you may want to execute commands from the text file in parallel. This procedure generates approximately 100k points for each model, and we have to downsample the models for evaluation:

```bash
./downsample_ground_truth.sh 03001627
```
