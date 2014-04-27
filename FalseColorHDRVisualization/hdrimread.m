function img = hdrimread(filename)
%
%       img = hdrimread(filename)
%
%       This function reads from a file with name filename an HDR image, if
%       the format can not be opened, it tries to open it as it was an LDR
%       image using imread from MATLAB Image Processing Toolbox.
%
%       JPEG and JPEG2000 files passed as input are meant to be compressed
%       using respectively JPEG-HDR or HDR JPEG-2000. 
%
%        Input:
%           -filename: the name of the file to open
%
%        Output:
%           -img: the opened image
%
%     Copyright (C) 2011-13  Francesco Banterle
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

if(~exist('filename', 'var'))
    error('A filename with extension needs to be passed as input!');
end

%extension = %lower(fileExtension(filename));

%Radiance format can have different extensions: .hdr, .rgbe and .pic
%if((strcmpi(extension,'pic')==1)||(strcmpi(extension,'rgbe')==1))
    extension = 'hdr';
%end

img = [];

bLDR = 0;

switch extension
    
    %PIC-HDR format by Greg Ward
    case 'hdr'
        try
		 %Uncompressed RGBE Image
            img = read_rgbe(filename);  
        catch err  
            try 
                %RLE compressed image
                img = double(hdrread(filename));
            catch err
                disp('Warning: this .hdr/.pic file can not be read.');
            end
        end
        
    %Portable float map
    case 'pfm'
        try
            img = read_pfm(filename);
        catch
            disp('Warning: this .pfm file can not be read.');
        end
        
    case 'jp2'
        try
            img = HDRJPEG2000Dec(filename);
        catch err
            bLDR = 1;            
            disp('Warning: this .jpg2 file cannot be read as an HDR JPEG 2000 file.');
        end        
        
    case 'jpg'
        try
            img = JPEGHDRDec(filename);
        catch err
            bLDR = 1;
            disp('Warning: this .jpg file cannot be read as a JPEG-HDR file.');
        end
       
    otherwise %try to open as LDR image
        try
            bLDR = 1;
            img = ldrimread(filename);
        catch err
            disp(['Warning: this LDR format, ',extension,', is not supported.']);
        end
end

if(isempty(img))
    if(bLDR == 1)
        img = ldrimread(filename);
        disp(['Warning: this image, ', filename,', has been loaded as an LDR image.']);
    else
        error(['This image,',filename,', cannot be loaded with LDR or HDR readers.']);
    end
end

%Remove specials
img = RemoveSpecials(img);

end
