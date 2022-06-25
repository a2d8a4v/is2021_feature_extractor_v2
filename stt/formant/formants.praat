###########################################################################
#                                                                      	  #
#  Praat Script: PyToBI           			                          	  #
#  Copyright (C) 2019  Mónica Domínguez-Bajo - Universitat Pompeu Fabra   #
#																		  #
#    This program is free software: you can redistribute it and/or modify #
#    it under the terms of the GNU General Public License as published by #
#    the Free Software Foundation, either version 3 of the License, or    #
#    (at your option) any later version.                                  #
#                                                                         #
#    This program is distributed in the hope that it will be useful,      #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of       #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
#    GNU General Public License for more details.                         #
#                                                                         #
#    You should have received a copy of the GNU General Public License    #
#    along with this program.  If not, see http://www.gnu.org/licenses/   #
#                                                                         #
###########################################################################
##### MODULE 3 ############################################################
###### Acoustic feature annotation										  #
###########################################################################

clearinfo
form File
    text filename
    positive maxformant 5500
    real winlen 0.025
    positive preemph 50
endform

Read from file... filename$
To Formant (burg)... 0.01 5 maxformant$ winlen$ preemph$
List... no yes 6 no 3 no 3 no

# clean Menu
select all
Remove