# Чтобы все работало нужно : 
## Поставить PyAudio
```
sudo apt-get install python3-pyaudio
```
```
sudo apt-get install python3-pyaudio
```
## Если вылазят ошибки, то
```
sudo apt-get install portaudio19-dev
sudo apt-get install python3-all-dev
```
## Поставить Porcupine
```
sudo pip3 install pvporcupine
```
## Для запуска Rhino 
```
python3 Путь_к_файлу/rhino_demo_file.py --input_video_file_path Путь_к_видео --context_file_path Путь_к_файлу classdetect_linux.rnn
```
## Например:
```
python3 demo/python/rhino_demo_file.py --input_video_file_path "video1.mp4" --context_file_path ./resources/contexts/linux/classdetect_linux.rhn
```

