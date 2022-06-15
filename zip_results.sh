if [ $# -eq 0 ]; then 
    zip -r results.zip ./logging/*
elif [ "$1" = "model" ]; then
    zip -r results.zip ./logging/* -x ./logging/*/events.out.tfevents.*
else
    echo "If you want to exclude tensorboard object insert 'model'"
fi