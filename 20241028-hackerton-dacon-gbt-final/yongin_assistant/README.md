# Overview
see: [url](https://yb98.notion.site/12ccd80736dc8070b3d0ea7b4324a1a2)

## Installation

```bash
sudo chmod 777 ./startup.sh
source ./startup.sh
```

## Deployment command
    
```bash
poetry lock --no-update 
docker build -t yongin-assistant:latest .
```