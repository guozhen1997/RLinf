export PYTHONPATH=/path/to/RLinf:$PYTHONPATH
export EMBODIED_PATH=/path/to/RLinf/examples/embodiment

cd ${EMBODIED_PATH}
python train_reward_model.py \
    positive_dir=/path/to/RLinf/reward_data/positive \
    negative_dir=/path/to/RLinf/reward_data/negative \
    output_checkpoint=/path/to/toolkits/reward_model/checkpoints/reward_model.pt \
    backbone=resnet10 \
    image_key=base_camera \
    batch_size=128 \
    epochs=30 \
    visualize_positive=true \
    vis_output_dir=/path/to/RLinf/examples/embodiment/show


