% #external e_id.
% #external e_position.
% #external e_size.
% #external e_color.
% #external e_material.
% #external e_shape.

% % Scene properties
% object(ID) :- obj(0,ID,_,_,_,_,_,_), not e_id.
% position(ID,X,Y) :- obj(0,ID,_,_,_,_,X,Y), not e_position.
% has_size(ID,SIZE) :- obj(0,ID,SIZE,_,_,_,_,_), not e_size.
% has_color(ID,COLOR) :- obj(0,ID,_,COLOR,_,_,_,_), not e_color.
% has_material(ID,MATERIAL):- obj(0,ID,_,_,MATERIAL,_,_,_), not e_material.
% has_shape(ID,SHAPE) :- obj(0,ID,_,_,_,SHAPE,_,_), not e_shape.

left_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', X<X'.
right_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', X>=X'.
in_front_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', Y>Y'.
behind_of(ID,ID') :- position(ID,X,Y), position(ID',X',Y'), state(T',ID'), ID!=ID', Y<=Y'.

in_between_projection(ID1,ID2,ID3) :- position(ID1,X1,Y1), position(ID2,X2,Y2), position(ID3,X3,Y3),
                                      ID1!=ID2, ID1!=ID3, ID2!=ID3,
                                      X1 < X3, X3 < X2.
in_between_projection(ID2,ID1,ID3) :- in_between_projection(ID1,ID2,ID3).

in_between_bbox(ID1,ID2,ID3) :- position(ID1,X1,Y1), position(ID2,X2,Y2), position(ID3,X3,Y3),
                                ID1!=ID2, ID1!=ID3, ID2!=ID3,
                                X1 < X3, X3 < X2,
                                Y1 < Y3, Y3 < Y2.
in_between_bbox(ID1,ID2,ID3) :- position(ID1,X1,Y1), position(ID2,X2,Y2), position(ID3,X3,Y3),
                                ID1!=ID2, ID1!=ID3, ID2!=ID3,
                                X1 < X3, X3 < X2,
                                Y2 < Y3, Y3 < Y1.
in_between_bbox(ID2,ID1,ID3) :- in_between_bbox(ID1,ID2,ID3).

in_between_proper(ID1,ID2,ID3) :- position(ID1,X1,Y1), position(ID2,X2,Y2), position(ID3,X3,Y3),
                                  ID1!=ID2, ID1!=ID3, ID2!=ID3,
                                  A = @distToSegment((X3,Y3),(X1,Y1),(X2,Y2)),
                                  A < 2.

% Uniqueness rule/constraint
state(T+1,ID) :- unique(T), state(T,ID).
:- unique(T), state(T,ID), state(T,ID'), ID!=ID'.

% Spatial relation rules
state(T+1,ID) :- relate_left(T), state(T,ID'), left_of(ID,ID').
state(T+1,ID) :- relate_right(T), state(T,ID'), right_of(ID,ID').
state(T+1,ID) :- relate_front(T), state(T,ID'), in_front_of(ID,ID') .
state(T+1,ID) :- relate_behind(T), state(T,ID'), behind_of(ID,ID').

% Count rule
int(T+1,V) :- count(T), #count{ ID : state(T,ID) } = V.

% Exist rule
bool(T+1,yes) :- exist(T), state(T,ID).
bool(T+1,no) :- exist(T), not bool(T+1,yes).

% Filtering rules
state(T+1,ID) :- filter_large(T), state(T,ID), has_size(ID,large).
state(T+1,ID) :- filter_small(T), state(T,ID), has_size(ID,small).
state(T+1,ID) :- filter_gray(T), state(T,ID), has_color(ID,gray).
state(T+1,ID) :- filter_red(T), state(T,ID), has_color(ID,red).
state(T+1,ID) :- filter_blue(T), state(T,ID), has_color(ID,blue).
state(T+1,ID) :- filter_green(T), state(T,ID), has_color(ID,green).
state(T+1,ID) :- filter_brown(T), state(T,ID), has_color(ID,brown).
state(T+1,ID) :- filter_purple(T), state(T,ID), has_color(ID,purple).
state(T+1,ID) :- filter_cyan(T), state(T,ID), has_color(ID,cyan).
state(T+1,ID) :- filter_yellow(T), state(T,ID), has_color(ID,yellow).
state(T+1,ID) :- filter_metal(T), state(T,ID), has_material(ID,metal).
state(T+1,ID) :- filter_rubber(T), state(T,ID), has_material(ID,rubber).
state(T+1,ID) :- filter_sphere(T), state(T,ID), has_shape(ID,sphere).
state(T+1,ID) :- filter_cylinder(T), state(T,ID), has_shape(ID,cylinder).
state(T+1,ID) :- filter_cube(T), state(T,ID), has_shape(ID,cube).

% Query functions
size(T+1,SIZE) :- query_size(T), state(T,ID), has_size(ID,SIZE).
color(T+1,COLOR) :- query_color(T), state(T,ID), has_color(ID,COLOR).
material(T+1,MATERIAL) :- query_material(T), state(T,ID), has_material(ID,MATERIAL).
shape(T+1,SHAPE) :- query_shape(T), state(T,ID), has_shape(ID,SHAPE).

% Logical operators
state(T+1,ID) :- and(T,T'), state(T,ID), state(T',ID).

state(T+1,ID) :- or(T,T'), state(T,ID).
state(T+1,ID') :- or(T,T'), state(T',ID').

bool(T+1, yes) :- boolean_negation(T), bool(T, no).
bool(T+1, no) :- boolean_negation(T), not bool(T+1, yes).

% Same-attribute relations
state(T+1,ID') :- same_size(T), state(T,ID), has_size(ID,SIZE), has_size(ID',SIZE), ID!=ID'.
state(T+1,ID') :- same_color(T), state(T,ID), has_color(ID,COLOR), has_color(ID',COLOR), ID!=ID'.
state(T+1,ID') :- same_material(T), state(T,ID), has_material(ID,MATERIAL), has_material(ID',MATERIAL), ID!=ID'.
state(T+1,ID') :- same_shape(T), state(T,ID), has_shape(ID,SHAPE), has_shape(ID',SHAPE), ID!=ID'.

% % Integer comparison
bool(T+1,yes) :- equal_integer(T,T'), int(T,V), int(T',V'), V=V'.
bool(T+1,no) :- equal_integer(T,T'), not bool(T+1,yes).

bool(T+1,yes) :- less_than(T,T'), int(T,V), int(T',V'), V<V'.
bool(T+1,no) :- less_than(T,T'), not bool(T+1,yes).

bool(T+1,yes) :- greater_than(T,T'), int(T,V), int(T',V'), V>V'.
bool(T+1,no) :- greater_than(T,T'), not bool(T+1,yes).

% Attribute comparison
bool(T+1,yes) :- equal_size(T,T'), size(T,V), size(T',V'), V=V'.
bool(T+1,no) :- equal_size(T,T'), not bool(T+1,yes).

bool(T+1,yes) :- equal_color(T,T'), color(T,V), color(T',V'), V=V'.
bool(T+1,no) :- equal_color(T,T'), not bool(T+1,yes).

bool(T+1,yes) :- equal_material(T,T'), material(T,V), material(T',V'), V=V'.
bool(T+1,no) :- equal_material(T,T'), not bool(T+1,yes).

bool(T+1,yes) :- equal_shape(T,T'), shape(T,V), shape(T',V'), V=V'.
bool(T+1,no) :- equal_shape(T,T'), not bool(T+1,yes).

% Equal Predicates

% derive equivalence classes 
equivalence_class(SZ,C,M,S) :- has_size(ID,SZ), has_color(ID,C), has_shape(ID,S), has_material(ID,M).
n_equal_objects(N,SZ,C,M,S) :- equivalence_class(SZ,C,M,S), 
                               N = #count{ ID,SZ,C,S,M : has_size(ID,SZ), has_color(ID,C), has_shape(ID,S), has_material(ID,M) }.

equal_objects :- n_equal_objects(N,_,_,_,_), N > 1.

bool(T+1,yes) :- all_different(T), not equal_objects.
bool(T+1,no) :- all_different(T), not bool(T+1,yes).

bool(T+1,yes) :- two_equal(T), n_equal_objects(N,_,_,_,_), N >= 2.
bool(T+1,no) :- two_equal(T), not bool(T+1,yes).

bool(T+1,yes) :- exactly_two_equal(T), n_equal_objects(2,_,_,_,_).
bool(T+1,no) :- exactly_two_equal(T), not bool(T+1,yes).

bool(T+1,yes) :- three_equal(T), n_equal_objects(N,_,_,_,_), N >= 3.
bool(T+1,no) :- three_equal(T), not bool(T+1,yes).

bool(T+1,yes) :- exactly_three_equal(T), n_equal_objects(3,_,_,_,_).
bool(T+1,no) :- exactly_three_equal(T), not bool(T+1,yes).

bool(T+1,yes) :- four_equal(T), n_equal_objects(N,_,_,_,_), N >= 4.
bool(T+1,no) :- four_equal(T), not bool(T+1,yes).

bool(T+1,yes) :- exactly_four_equal(T), n_equal_objects(4).
bool(T+1,no) :- exactly_four_equal(T), not bool(T+1,yes).

all(A) :- A = #count{ ID : object(ID) }, set_difference(T).
int(T+1,X) :- set_difference(T), all(A), int(T,B), X = A - B.

state(T,ID) :- count_different(T), state(T,ID).
int(T+1,X) :-  count_different(T), 
               X = #count{ ID : object(ID)}.

% Between Predicates

% Between projection
bool(T+1, yes) :- between_projection(T,T',T''), 
                  state(T,ID), state(T',ID'), state(T'',ID''), 
                  in_between_projection(ID,ID',ID'').

bool(T+1, no) :- between_projection(T,T',T''), not bool(T+1,yes).

% Between bbox
bool(T+1, yes) :- between_bbox(T,T',T''), 
                  state(T,ID), state(T',ID'), state(T'',ID''), 
                  in_between_bbox(ID,ID',ID'').

bool(T+1, no) :- between_bbox(T,T',T''), not bool(T+1,yes).

% Between proper
bool(T+1, yes) :- between_proper(T,T',T''), 
                  state(T,ID), state(T',ID'), state(T'',ID''), 
                  in_between_proper(ID,ID',ID'').

bool(T+1, no) :- between_proper(T,T',T''), not bool(T+1,yes).

% Count Between Predicates

between_projection_count(T,T',0):- count_between_projection(T,T').
int(T+1,A) :- A = #count{ ID : state(T+1,ID) }, count_between_projection(T,T').
state(T+1,ID'') :- between_projection_count(T,T',T''), 
                   state(T,ID), state(T',ID'), state(T'',ID''),
                   in_between_projection(ID,ID',ID'').

between_bbox_count(T,T',0):- count_between_bbox(T,T').
int(T+1,A) :- A = #count{ ID : state(T+1,ID) }, count_between_bbox(T,T').
state(T+1,ID'') :- between_bbox_count(T,T',T''),
                   state(T,ID), state(T',ID'), state(T'',ID''),
                   in_between_bbox(ID,ID',ID'').

between_proper_count(T,T',0):- count_between_proper(T,T').
int(T+1,A) :- A = #count{ID : state(T+1,ID) }, count_between_proper(T,T').
state(T+1,ID'') :- between_proper_count(T,T',T''), 
                   state(T,ID), state(T',ID'), state(T'',ID''),
                   in_between_proper(ID,ID',ID'').

% State rules
state(0,ID) :- object(ID).
state(T+1,ID) :- scene(T), object(ID).

% Derive answer (T must equal the last point in time)
ans(V) :- end(T), size(T,V).
ans(V) :- end(T), color(T,V).
ans(V) :- end(T), material(T,V).
ans(V) :- end(T), shape(T,V).
ans(V) :- end(T), bool(T,V).
ans(V) :- end(T), int(T,V).

:- not ans(_).

#show ans/1.
% #show state/2.
