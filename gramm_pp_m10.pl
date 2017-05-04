s(X) --> np(X), vp(X).
vp(X) --> v(X), np(X).
np(X) --> det(X), n(X),{random(0,2,X)}.
np(X) --> adj(X), n(X),{random(0,1,X)}.
pp(X) --> ins(X), np(X).
%np(X) --> np(X), pp(X),{random(0,3,X)}.
np(X) --> ['john'],{random(0,4,X)}.
np(X) --> ['jim'],{random(0,4,X)}.
np(X) --> ['tradition'],{random(0,4,X)}.
np(X) --> ['premarin'],{random(0,4,X)}.
np(X) --> ['whimsy'],{random(0,4,X)}.
np(X) --> ['coatings'],{random(0,4,X)}.
np(X) --> ['rescissions'],{random(0,4,X)}.
np(X) --> ['retailing'],{random(0,4,X)}.
np(X) --> ['cityfed'],{random(0,4,X)}.
np(X) --> ['importers'],{random(0,4,X)}.
np(X) --> ['samuel'],{random(0,4,X)}.
np(X) --> ['t-shirts'],{random(0,4,X)}.
np(X) --> ['integration'],{random(0,4,X)}.
np(X) --> ['bookkeeper'],{random(0,4,X)}.
np(X) --> ['cynthia'],{random(0,4,X)}.
np(X) --> ['passers-by'],{random(0,4,X)}.
np(X) --> ['tampons'],{random(0,4,X)}.
np(X) --> ['franco'],{random(0,4,X)}.
np(X) --> ['masons'],{random(0,4,X)}.
np(X) --> ['vineyard'],{random(0,4,X)}.
n(X) --> ['boy'],{random(0,4,X)}.
n(X) --> ['fish'],{random(0,4,X)}.
n(X) --> ['dog'],{random(0,4,X)}.
n(X) --> ['cat'],{random(0,4,X)}.
n(X) --> ['girl'],{random(0,4,X)}.
n(X) --> ['addiction'],{random(0,4,X)}.
n(X) --> ['cable-tv-system'],{random(0,4,X)}.
n(X) --> ['fret'],{random(0,4,X)}.
n(X) --> ['cycle'],{random(0,4,X)}.
n(X) --> ['stop-motion'],{random(0,4,X)}.
n(X) --> ['casting'],{random(0,4,X)}.
n(X) --> ['candidate'],{random(0,4,X)}.
v(X) --> ['sees'],{random(0,4,X)}.
v(X) --> ['like'],{random(0,4,X)}.
v(X) --> ['see'],{random(0,4,X)}.
v(X) --> ['eat'],{random(0,4,X)}.
v(X) --> ['fishing'],{random(0,4,X)}.
v(X) --> ['eating'],{random(0,4,X)}.
v(X) --> ['ate'],{random(0,4,X)}.
v(X) --> ['likes'],{random(0,4,X)}.
v(X) --> ['overreacting'],{random(0,4,X)}.
v(X) --> ['insured'],{random(0,4,X)}.
v(X) --> ['quips'],{random(0,4,X)}.
v(X) --> ['set'],{random(0,4,X)}.
v(X) --> ['reappraised'],{random(0,4,X)}.
v(X) --> ['coming'],{random(0,4,X)}.
v(X) --> ['fled'],{random(0,4,X)}.
v(X) --> ['specified'],{random(0,4,X)}.
v(X) --> ['jailed'],{random(0,4,X)}.
v(X) --> ['oppose'],{random(0,4,X)}.
v(X) --> ['centralized'],{random(0,4,X)}.
adj(X) --> ['big'],{random(0,10,X)}.
adj(X) --> ['small'],{random(0,10,X)}.
adj(X) --> ['avaricious'],{random(0,10,X)}.
adj(X) --> ['ever-faster'],{random(0,10,X)}.
adj(X) --> ['value-oriented'],{random(0,10,X)}.
adj(X) --> ['church-owned'],{random(0,10,X)}.
adj(X) --> ['lowest-rated'],{random(0,10,X)}.
adj(X) --> ['besieged'],{random(0,10,X)}.
adj(X) --> ['ever-greater'],{random(0,10,X)}.
adj(X) --> ['anti-galileo'],{random(0,10,X)}.
adj(X) --> ['limited-edition'],{random(0,10,X)}.
adj(X) --> ['substantial'],{random(0,10,X)}.
adj(X) --> ['now-evident'],{random(0,10,X)}.
adj(X) --> ['behavioral'],{random(0,10,X)}.
adj(X) --> ['cancerous'],{random(0,10,X)}.
adj(X) --> ['raw-materials'],{random(0,10,X)}.
adj(X) --> ['polish'],{random(0,10,X)}.
adj(X) --> ['more-affordable'],{random(0,10,X)}.
det(X) --> ['every'],{random(0,7,X)}.
det(X) --> ['nary'],{random(0,7,X)}.
det(X) --> ['no'],{random(0,7,X)}.
det(X) --> ['another'],{random(0,7,X)}.
det(X) --> ['either'],{random(0,7,X)}.
det(X) --> ['del'],{random(0,7,X)}.
det(X) --> ['the'],{random(0,7,X)}.
det(X) --> ['some'],{random(0,7,X)}.
det(X) --> ['a'],{random(0,7,X)}.
det(X) --> ['these'],{random(0,4,X)}.
ins(X) --> ['at'],{random(0,10,X)}.
ins(X) --> ['for'],{random(0,10,X)}.
ins(X) --> ['in'],{random(0,10,X)}.
ins(X) --> ['lest'],{random(0,10,X)}.
ins(X) --> ['including'],{random(0,10,X)}.
ins(X) --> ['during'],{random(0,10,X)}.
ins(X) --> ['out'],{random(0,10,X)}.
ins(X) --> ['far'],{random(0,10,X)}.
ins(X) --> ['are'],{random(0,10,X)}.
ins(X) --> ['atop'],{random(0,10,X)}.
ins(X) --> ['in'],{random(0,10,X)}.
ins(X) --> ['near'],{random(0,10,X)}.
ins(X) --> ['post'],{random(0,10,X)}.
ins(X) --> ['into'],{random(0,10,X)}.
ins(X) --> ['up'],{random(0,10,X)}.



genera_frasi([]).
genera_frasi([X|R]):-
	s(X,S,[]),
	write('"'),writesentence(S),write('"'),nl,
	genera_frasi(R).
	
writesentence([]).
writesentence([A|L]):-
	write(A), write(' '),
	writesentence(L).
