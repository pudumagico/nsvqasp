% search space 
material(metal;rubber).
size(small;large).
shape(sphere;cylinder;cube).
color(brown;yellow;green;blue;purple;cyan;red;gray).
width_pixel(120;240;360).
height_pixel(80;160;240).

changeable(ID) :- conf(ID,C), C < 90.

obj(ID) :- obj(0,ID,_,_,_,_,_,_), not changeable(ID).
position(ID,X,Y) :- obj(0,ID,_,_,_,_,X,Y), obj(ID), not changeable(ID).
has_size(ID,SZ) :- obj(0,ID,SZ,_,_,_,_,_), obj(ID), not changeable(ID).
has_color(ID,C) :- obj(0,ID,_,C,_,_,_,_), obj(ID), not changeable(ID).
has_material(ID,M) :- obj(0,ID,_,_,M,_,_,_), obj(ID), not changeable(ID).
has_shape(ID,S) :- obj(0,ID,_,_,_,S,_,_), obj(ID), not changeable(ID).

% guess a change for the object or omit it entirely
{ object(ID) } :- obj(0,ID,_,_,_,_,_,_), changeable(ID).
{ position(ID,X+(D*10),Y+(E*10)) : D=-1..1, E=-1..1 } = 1 :- object(ID), obj(0,ID,_,_,_,_,X,Y), changeable(ID).
{ position(ID,X,Y) : width_pixel(X), height_pixel(Y) } = 1 :- object(ID), not obj(0,ID,_,_,_,_,_,_), changeable(ID).
{ has_color(ID,C) : color(C) } = 1 :- object(ID), changeable(ID).
{ has_size(ID,SZ) : size(SZ) }  = 1 :- object(ID), changeable(ID).
{ has_material(ID,M) : material(M) } = 1 :- object(ID), changeable(ID).
{ has_shape(ID,S) : shape(S) } = 1 :- object(ID), changeable(ID).

% additional objects (limited to 3)
{ add_objects(0..3) } = 1.
{ object(M+1..M+N) } = N :- add_objects(N), M = #max{ X : obj(0,X,_,_,_,_,_,_) }.

% check that the expected answer is derived
:- ans(A), #count{ 1 : foil(A) } != 1.

% derive concrete changes for better readability
size_change(ID,SZ,SZ') :- has_size(ID,SZ'), obj(0,ID,SZ,_,_,_,_,_), SZ!=SZ'.
color_change(ID,C,C') :- has_color(ID,C'), obj(0,ID,_,C,_,_,_,_), C!=C'.
material_change(ID,M,M') :- has_material(ID,M'), obj(0,ID,_,_,M,_,_,_), M!=M'.
shape_change(ID,SH,SH') :- has_shape(ID,SH'), obj(0,ID,_,_,_,SH,_,_), SH!=SH'.
moved_left(ID,X,Y,X',Y') :- position(ID,X',Y'), obj(0,ID,_,_,_,_,X,Y), X>X'.
moved_right(ID,X,Y,X',Y') :- position(ID,X',Y'), obj(0,ID,_,_,_,_,X,Y), X<X'.
moved_down(ID,X,Y,X',Y') :- position(ID,X',Y'), obj(0,ID,_,_,_,_,X,Y), Y>Y'.
moved_up(ID,X,Y,X',Y') :- position(ID,X',Y'), obj(0,ID,_,_,_,_,X,Y), Y<Y'.
object_omission(ID) :- obj(0,ID,_,_,_,_,_,_), not object(ID), changeable(ID).
object_addition(ID,SZ,C,M,S,X,Y) :- not obj(0,ID,_,_,_,_,_,_), object(ID),
                                    has_size(ID,SZ), has_color(ID,C),
                                    has_material(ID,M), has_shape(ID,S),
                                    position(ID,X,Y).

% minimize changes and ommissions
:~ moved_up(ID,X,Y,X',Y'), conf(ID,C). [100-C+10,ID,movement]
:~ moved_down(ID,X,Y,X',Y'), conf(ID,C). [100-C+10,ID,movement]
:~ moved_left(ID,X,Y,X',Y'), conf(ID,C). [100-C+10,ID,movement]
:~ moved_right(ID,X,Y,X',Y'), conf(ID,C). [100-C+10,ID,movement]
:~ size_change(ID,SZ,SZ'), conf(ID,C). [100-C+11,ID,size_change]
:~ color_change(ID,C,C'), conf(ID,C). [100-C+11,ID,color_change]
:~ material_change(ID,M,M'), conf(ID,C). [100-C+11,ID,material_change]
:~ shape_change(ID,SH,SH'), conf(ID,C). [100-C+11,ID,shape_change]
:~ object_omission(ID), conf(ID,C). [100-C+12,ID,object_omission]
:~ object_addition(ID,SZ,C,M,S,X,Y). [13,ID,object_addition]

#show size_change/3.
#show color_change/3.
#show material_change/3.
#show shape_change/3.
#show object_omission/1.
#show object_addition/7.
#show moved_down/5.
#show moved_up/5.
#show moved_left/5.
#show moved_right/5.
#show changeable/1.
% #show conf/2.
% #show object/1.
% #show position/3.