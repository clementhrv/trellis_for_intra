for i in {1..5}
do
    python -m classification_latents.train \
                --training_parameters_path=classification_latents/classification.json \
                --project_name='part14_test_PNP200_run'$i \
                --project_folder='trellis_test'\
                --num_epochs=200 \
                --init_lr=0.001 \
                --batch_size=16 \
                --warmup=500\
                --num_workers=0 \
                --prefetch_factor=0 \
                --model_save_path=model.ckpt
done
