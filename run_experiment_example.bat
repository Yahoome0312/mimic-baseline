@echo off
echo ================================================================================
echo ISIC 2019 CLIP Training - Custom Experiment Name Example
echo ================================================================================
echo.
echo This example shows how to use custom experiment names and output directories.
echo.
echo Experiment: Weighted CLIP Loss with custom learning rate
echo Output: results/weighted_lr5e6
echo Files will be saved with prefix: weighted_inverse_lr5e6_exp1
echo.
pause

python main.py ^
    --method finetune ^
    --loss_type weighted ^
    --class_weight_method inverse ^
    --lr_image 5e-6 ^
    --lr_text 1e-4 ^
    --output_dir results/weighted_lr5e6 ^
    --experiment_name weighted_inverse_lr5e6_exp1

echo.
echo ================================================================================
echo Training completed!
echo ================================================================================
echo.
echo Results saved to: results\weighted_lr5e6\
echo.
echo Generated files:
echo   - weighted_inverse_lr5e6_exp1_best_model.pth
echo   - weighted_inverse_lr5e6_exp1_training_curves.png
echo   - weighted_inverse_lr5e6_exp1_results.json
echo   - weighted_inverse_lr5e6_exp1_confusion_matrix.png
echo   - weighted_inverse_lr5e6_exp1_per_class_recall.png
echo.
pause
