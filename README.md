**Projet 4 - Groupe 1**

> Cinthya (CdP), Marouan, Nolan


**Table of contents**

[TOC]

## Description de l'application

The Mask-Arade est une application FastApi qui a pour but de détecter le port du masque sur les personnes présentes à Simplon Grenoble sur une image ou une vidéo.

Cette application utilise en arrière plan un algorithme de classification qui classifie les images en 3 catégories :

- **Correct** : port correct du masque.
- **Incorrect** : port incorrect du masque (sous le nez, sur le menton...).
- **Missing** : absence du masque.

Un autre algorithme de détection de visage est utilisé pour déterminer si un ou des visages sont présents sur l'image et effectuer une classification sur chacun des visages détectés.

Elle comporte 3 pages web proposant chacune un support différent pour effectuer la classification :

- **Upload File** : classification sur un fichier image uploadé (fonctionne avec plusieurs personnes sur une seule image).
- **Take picture** : prise d'une photo avec la camera de la machine et classification sur l'image obtenue (fonctionne avec une seule personne).
- **Video** : classification en temps réel sur la vidéo issue de la webcam de la machine (fonctionne avec une seule personne).

<img src="git_web.gif">

## Installation et lancement de l'application

Cloner le dépôt git 
```bash
git clone url_repo
```

Créer l'environnement virtuel et installer les dépendances
```bash
pipenv install
```

Entrer dans l'environnement virtuel
```bash
pipenv shell
```

Lancer l'application
```bash
uvicorn main:app --reload
```

## Modèles utilisés

**Détection de visage** : mediapipe.

**Classification sur la page Upload File** : Random Forest.

**Classification sur les pages Take picture et Video** : tensorflow.js.

## Lien vers l'application en ligne

https://maskdetector.azurewebsites.net/


