python scripts/training/evaluate_critics_live.py \
  -c outputs/checkpoints/critic_only/realman_itqc_reward/1:1_mc_0.8/critic_only_ckpt.pth \
   --config outputs/checkpoints/critic_only/realman_itqc_reward/1:1_mc_0.8/config_1:1.yaml \
   -d datasets/realman/Stack_SF/Stack_SF.h5\
   -w 200
