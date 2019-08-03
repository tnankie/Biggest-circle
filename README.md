# Biggest-circle
currently works on simplified tracklog
intent is to take an .igc tracklog test if it is "closed" (proximity of initial point to final point is within 20% of calculated circle circumfrence)
then calculate the largest circle enclosed by the tracklog.

Reason: to score local paragliding flights and award a silly prize for whoever does the largest diameter flight of the year.

Polygon code taken from https://github.com/Twista/python-polylabel
which is based on https://github.com/mapbox/polylabel
using the paper https://sites.google.com/site/polesofinaccessibility/ and http://web.archive.org/web/20140629230429/http://cuba.ija.csic.es/~danielgc/papers/Garcia-Castellanos,%20Lombardo,%202007,%20SGJ.pdf
