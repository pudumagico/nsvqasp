% filename CLEVR_val_000107.png
% scene obj(0,0,large,blue,rubber,cylinder,182,215).obj(0,1,large,gray,rubber,cube,285,114).obj(0,2,small,green,metal,sphere,365,162).obj(0,3,large,blue,rubber,cube,261,72).obj(0,4,small,blue,rubber,sphere,74,144).
% question boolean_negation(16,16).between_bbox(15,15,9,4).unique(14,14).filter_sphere(13,13).filter_rubber(12,12).filter_blue(11,11).filter_small(10,10).scene(9,0).unique(8,8).filter_cylinder(7,7).filter_metal(6,6).filter_gray(5,5).scene(4,0).unique(3,3).filter_gray(2,2).filter_large(1,1).scene(0,0).end(17).
% predicted answer yes
% ground truth scene obj(0,0,large,blue,rubber,cylinder,183,216).obj(0,1,small,green,metal,sphere,365,162).obj(0,2,small,blue,rubber,sphere,74,144).obj(0,3,large,blue,rubber,cube,261,72).obj(0,4,large,gray,rubber,cube,284,114).obj(0,5,small,gray,metal,cylinder,325,113).obj(0,6,small,green,rubber,sphere,135,199).
% ground truth question boolean_negation(16,16).between_bbox(15,15,9,4).unique(14,14).filter_sphere(13,13).filter_rubber(12,12).filter_blue(11,11).filter_small(10,10).scene(9,0).unique(8,8).filter_cylinder(7,7).filter_metal(6,6).filter_gray(5,5).scene(4,0).unique(3,3).filter_gray(2,2).filter_large(1,1).scene(0,0).end(17).
% ground truth answer no

obj(0,0,large,yellow,metal,sphere,343,164).
obj(0,1,large,purple,metal,cylinder,96,133).
obj(0,2,large,purple,rubber,cube,226,199).
obj(0,3,large,green,rubber,cube,187,105).
obj(0,4,small,green,metal,cylinder,299,101).
foil(no).

scene(0).
filter_small(1).
unique(2).
same_color(3).
exist(4).
end(5).