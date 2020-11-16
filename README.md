# Configuração

## Virtualenv ( Python 3 )

`python -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements.txt`

# Uso

## Para ler a entrada de vídeo de um arquivo:

```
python crowdcontrol.py --input videos/example_01.mp4 -output output/output_01.avi
```

##  Para ler a entrada de vídeo de uma webcam:

```
python crowdcontrol.py --output output/webcam_output.avi
```
