rm -r ../build

mkdir ../build

cd ../build

cmake ..

make

cd ../scripts

sh run.sh