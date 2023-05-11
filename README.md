# LandMark-Recognition

## Содержание
  * [Введение](https://github.com/bitxzibit3/LandMark-Recognition/edit/main/README.md#%D0%B2%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5)
  * [Установка зависимостей](https://github.com/bitxzibit3/LandMark-Recognition/edit/main/README.md#%D1%83%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-%D0%B7%D0%B0%D0%B2%D0%B8%D1%81%D0%B8%D0%BC%D0%BE%D1%81%D1%82%D0%B5%D0%B9)
  * [Получение предсказаний](https://github.com/bitxzibit3/LandMark-Recognition/edit/main/README.md#%D0%BF%D0%BE%D0%BB%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5-%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B9)

## Введение
  Данный проект посвящен теме распознавания достопримечательностей на фотографиях. Для решения этой задачи использовались нейросетевые модели. Значения метрик, функций потерь, а также гиперпараметры и файлы, связанные с весами сетей, можно найти по [ссылке](https://wandb.ai/ml_landmarks/ml_landmarks).

## Установка зависимостей
После клонирования репозитория, вы можете установить необходимые зависимости, воспользовавшись следующей командой:
```bash
pip3 install -r requirements.txt
```

## Получение предсказаний
Для получения предсказаний нужно запустить файл `predict.py`. В качестве аргументов в него необходимо передать:
  * путь до файла с изображением
  * вид сети, который вы хотите использовать. Сейчас доступно две сети: 
    * VGG13 с батч-нормализацией, классификатор который был дообучен под данную задачу.
    * Сеть с архитектурой, представленной в [данном файле](https://github.com/bitxzibit3/LandMark-Recognition/blob/main/models/my_model.py).
  * также, в качестве необязательного аргумента вы можете передать устройство, на котором вы бы хотели проводить вычисления.
 В качестве ответа вы получите строку с потенциальным названием достопримечательности. С доступными классами вы можете ознакомиться в [данном файле](https://github.com/bitxzibit3/LandMark-Recognition/blob/main/data/classes.txt).