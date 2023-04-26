python ./train_sketch.py --sketch-datadir "/root/autodl-tmp/SHREC13/13_sketch_train_picture" --val-sketch-datadir "/root/autodl-tmp/SHREC13/13_sketch_test_picture" --num-classes 90 --lr-backbone 8e-4 --lr-classifier 8e-3 --sketch-batch-size 128 --loss "sketchmag" --margin 0.75 --sem_margin 0 --easy_margin false --model-dir "./saved_models/Shrec13/ResNet50_qualitymag"