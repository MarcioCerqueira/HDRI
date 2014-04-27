function stack = ReadLDRStack(dir_name, format)
%
%       stack = ReadLDRStack(dir_name, format)
%
%
%        Input:
%           -dir_name: the path where the stack is
%           -format: the LDR format of the images that we want to load in
%           the folder dir_name.
%
%        Output:
%           -stack: a stack of LDR images, in floating point (single)
%           format. No normalization is applied.
%
%     This function reads a stack of images from the disk
%
%     Copyright (C) 2011  Francesco Banterle
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

list = dir([dir_name,'/*.',format]);
n = length(list);

if(n>1)
    info = imfinfo([dir_name,'/',list(1).name]);
    stack = zeros(info.Height, info.Width, 3, n);

    for i=1:n
        disp(list(i).name);
        %read an image, and convert it into floating-point
        img = single(imread([dir_name,'/',list(i).name]));  

        %store in the stack
        stack(:,:,:,i) = img;    
    end
else
    disp('The stack is empy!');
    stack = [];
end

end
