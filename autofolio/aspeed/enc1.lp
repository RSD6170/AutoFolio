#script(python)

from clingo import Number, Tuple_, Function
from clingo.symbol import parse_term

ts = {}
def insert(i,s,t):
  key = str(s)
  if not ts.get(key):
    ts[key] = []
  ts[key].append([i,t])
  return parse_term("1")

def order(s):
  key = str(s)
  if not ts.get(key):
    ts[key] = []
  ts[key].sort(key=lambda x: int(x[1].number))
  p = None
  r = []
  for i, v in ts[key]:
    if p:
      r.append(Tuple_([p,i]))
    p = i
  return Tuple_(r)

#end.

#const cores=1.

solver(S)  :- time(_,S,_).
time(S,T)  :- time(_,S,T).
unit(1..cores).

insert(@insert(I,S,T)) :- time(I,S,T).
order(I,K,S) :- insert(_), solver(S), (I,K) = @order(S).

{ slice(U,S,T) : time(S,T), T <= K, unit(U) } 1 :-
  solver(S), kappa(K).
slice(S,T) :- slice(_,S,T).

 :- not #sum { T,S : slice(U,S,T) } K, kappa(K), unit(U).

solved(I,S) :- slice(S,T), time(I,S,T).
solved(I,S) :- solved(J,S), order(I,J,S).
solved(I)   :- solved(I,_).

#maximize { 1@2,I: solved(I) }.
#minimize { T*T@1,S : slice(S,T)}.

#show slice/3.
