function [] = plot_csv( label_file, xyz_file )
%PLOT_CSV Plots the Python csv output
labels = csvread(label_file);
xyz = csvread(xyz_file);
plotLabeledPointcloud(xyz, labels)

end

