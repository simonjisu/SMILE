if [ $# -eq 0 ]; then 
    python main.py --exp $1
elif [ "$1" = "nohup" ]; then
    nohup python -u main.py --exp $1 > $1.log &
else
    echo "If you want to run train with nohup, please insert 'nohup' as arugment"
fi

