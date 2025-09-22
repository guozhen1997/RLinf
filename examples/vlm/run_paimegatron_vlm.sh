pushd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash run_mcore_qwen.sh  \
	dsw  \
	3B   \
	1    \
	32 \
	1e-5   \
	1e-6   \
	2048  \
	2048  \
	bf16  \
	2   \
	2  \
	1 \
	true \
	true \
	true   \
	none \
	false \
	100000  \
	/mnt/public/tangyueran/llava-datasets-wds/wds   \
	/mnt/public/tangyueran/llava-datasets-wds/wds   \
	/mnt/public/tangyueran/Qwen2.5-VL-3B-Instruct \
	20000  \
	200   \
	./output_mcore_qwen2_5_vl_pretrain

popd
