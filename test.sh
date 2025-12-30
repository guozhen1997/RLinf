export ROBOTWIN_PATH="/mnt/public/peihong/codes/Robotwin_support"

export ROBOT_PLATFORM=ALOHA # $ROBOT_PLATFORM
export PYTHONPATH="/mnt/public/peihong/codes/RLinf:/opt/libero:${ROBOTWIN_PATH}:/mnt/public/peihong/codes/RLinf:/opt/libero:$PYTHONPATH"
python tests/unit_tests/test_robotwin_env.py
# python tests/unit_tests/test_model.py
