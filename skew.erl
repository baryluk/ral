-module(skew).
-author('baryluk@smp.if.uj.edu.pl').

-export([zero/0, inc/1, to_number/1, to_number2/1, to_display/1, first/1]).
-export([a_zero/0, a_inc/1, a_number/1]).

% sparse skew binary number representation
% See http://pl.wikipedia.org/wiki/Skośny_system_dwójkowy


% 0.
zero() -> [].

% inc increments number by 1 in O(1).
inc([X,X|T]) -> [X+1|T];
inc([X|T])   -> [0,X|T];
inc([])      -> [0].

% [2,2,3,5] -> "101200".
to_display([]) -> [$0];
to_display(L)  -> to_display(L, [], false, 0).

to_display([X,X|T], Acc, _, X)  -> to_display(T, [$2 | Acc], false, X+1);
to_display([X|T], Acc, _, X)    -> to_display(T, [$1 | Acc], false, X+1);
to_display([_|_T]=L, Acc, _, X) -> to_display(L, [$0 | Acc], true, X+1);
to_display([], Acc, true, _X)   -> [$1 | Acc];
to_display([], Acc, false, _X)  -> Acc.

% [2,2,3,5] -> [7,7,15,63].
to_exps(L) -> [ (1 bsl (X+1)) - 1 || X <- L ].
% [7,7,15,63] -> "7+7+15+63".
to_sum([]) -> "0";
to_sum(L) -> string:join([ integer_to_list(P) || P <- to_exps(L) ], "+").
%[E1|T] = to_exps(L), lists:foldl(fun(P, Acc) -> integer_to_list(P) ++ [$+|Acc] end, integer_to_list(E1), T).


% [2,2,3,5] -> 92.
to_number(L)  -> lists:foldl(fun(X, Acc) -> Acc +  ((1 bsl (X+1)) - 1) end, 0, L).

% alternative version
to_number2(L) -> lists:sum(to_exps(L)).

% show first N numbers
first(N) -> lists:foldl(fun(I, X) ->
		io:format("  ~p:   \t~p     \t= ~s        \t= ~p     \t= ~s_skew~n", [I-1, X, to_sum(X), to_number(X), to_display(X)]),
		inc(X)
	end, zero(), lists:seq(1, N)).


% other representation of numbers

a_zero() -> [].

a_inc([X,X|T]) -> [(X bsl 1)+1|T];
a_inc([H|T]) ->   [1,H|T];
a_inc([]) ->      [1].

a_number(L) -> lists:sum(L).

%lists:foldl(fun(I, X) -> io:format("~p:   \t~p  =  \t~p~n", [I-1, X, ral:a_number(X)]), ral:a_inc(X) end, [], lists:seq(1,100)).
