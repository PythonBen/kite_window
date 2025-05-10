# Affichage de la fenetre de vol d'un kite

L'utilisateur entre la vitesse du vent réel, celui ci vient du Nord (ou du haut de la figure).

Il entre ensuite, l'angle de la trajectoire. C'est l'angle définit par rapport à l'horizontale. Ainsi 0 degree correspond à une direction plein Est. 45 degré correspond à une direction Nord Est, 90 degré correspond à une direction plein Nord, ect.

Enfin, il faut entrer la vitesse du rider, ce qui permet de calculer le vent apparent et
d'afficher la fenetre.

Pour plus d'info, voir le code python est dans le fichier main.
On a utilisé la librairie fasthtml ainsi que matplotlib.

Remarques: En pratique, plus le vent est léger et plus la limite avant recule. La limite 
avant dépend aussi du kite, par exemple les ailes à caisson à haut ratio ont une limite avant plus proche de la limite théorique que celle d'une aile à boudin. 

