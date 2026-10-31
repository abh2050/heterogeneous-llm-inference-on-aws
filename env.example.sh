export AWS_PROFILE=mgmt
export AWS_REGION=us-east-2
export MODEL=Qwen/Qwen2.5-1.5B-Instruct
export VPC_ID=vpc-REPLACE
export SG_ID=sg-REPLACE
export SSH_KEY=~/.ssh/vllm-hetero-mgmt.pem
export VLLM_VERSION=0.29.0
export CPU_IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.29.0
# Home IP changed? Re-authorize SSH:
#   aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32
