python ./train_sketch.py --sketch-datadir "/root/autodl-tmp/SHREC13/13_sketch_train_picture" --val-sketch-datadir "/root/autodl-tmp/SHREC13/13_sketch_test_picture" --model "resnest50" --num-classes 90 --lr-backbone 4e-4 --lr-classifier 4e-3 --sketch-batch-size 64 --loss "sketchmag" --margin 0.75 --sem_margin 0.6 --wd 2e-5 --easy_margin false --model-dir "./saved_models/Shrec13/ResNest50_sketchmag" --max-epoch 200