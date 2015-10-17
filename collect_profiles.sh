mkdir tmp;
mkdir profiles/$1;
mv r000ah tmp;
amplxe-cl -report hotspots -r ./tmp/r000ah/ > profiles/$1/profile.txt;
amplxe-cl -report hotspots -source-object function="Central2D<Shallow2D, MinMod<float>>::limited_derivs" -r ./tmp/r000ah/ > profiles/$1/limited_derivs-profile.txt;
amplxe-cl -report hotspots -source-object function="Central2D<Shallow2D, MinMod<float>>::compute_step" -r ./tmp/r000ah/ > profiles/$1/compute_step-profile.txt;
amplxe-cl -report hotspots -source-object function="Central2D<Shallow2D, MinMod<float>>::compute_fg_speeds" -r ./tmp/r000ah/ > profiles/$1/compute_fg_speeds-profile.txt;
mv ipo_out.optrpt profiles/$1;
rm -rf tmp;
