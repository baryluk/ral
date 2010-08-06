-module(ral).
-author('baryluk@smp.if.uj.edu.pl').

% Copyright: Witold Baryluk, 2010
% License: BSD

-export([empty/0, is_empty/1, head/1, nth/2, tail/1, cons/2, length/1, from_list/1]).
-export([to_list/1, nthtail/2]).
-export([to_list_reversed/1, from_list_reversed/1]).
-export([last/1, foldl/3, foldr/3, map/2, foreach/2, mapfoldl/3, mapfoldr/3, dropwhile/2]).
-export([foldl_cancelable/3]).
-export([reverse_list/2]).
-export([replace/3]).

-compile({no_auto_import, [length/1]}).

-ifdef(TEST).
-include_lib("eunit/include/eunit.hrl").

-define(t(X), ?_test(?assert(X))).

general_lists() ->
	[
		[],
		[1],
		[3,4],
		[2,4,6],
		[2,4,6,85],
		[2,4,6,85,2],
		[2,4,6,85,2,6],
		[2,4,6,85,2,6,1],
		[2,4,6,85,2,6,1,2],
		[2,4,6,85,2,6,1,6,7],
		[2,4,6,85,2,6,1,2,6,7],
		[2,4,6,85,2,6,1,6,7,92,1,4],
		[2,4,6,85,2,6,1,2,6,7,92,1,4],
		[2,4,6,87,1,2,6,1,2,4,6,2,35,2,3,15,423,1234,12,46,123]
		].

general_testing1(F0) when is_function(F0, 2) ->
	% F0(L, from_list(L))
	NLs = general_lists(),
	lists:map(fun(NL) ->
			RAL = from_list(NL),
			F = fun() -> F0(RAL, NL) end,
			?_test(F())
		end,
		NLs).

general_testing2(F0) when is_function(F0, 3) ->
	% F0(L, from_list(L), Y)
	NLs = general_lists(),
	lists:map(fun(NL) ->
			RAL = from_list(NL),
			F = fun(Y) -> F0(RAL, NL, Y) end,
			[ ?_test(F(T)) || T <- [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 35, 46, 123, 9999] ]
		end,
		NLs).
-endif.

% random-access-list implementation

empty() ->
	[].

-ifdef(TEST).
query_test_fun_gen(RAL_QUERY, LIST_QUERY) ->
	fun(RAL, NL) ->
		AR = RAL_QUERY(RAL),
		A = LIST_QUERY(NL),
		?assert(AR =:= A),
		ok
	end.
-endif.


is_empty([]) ->
	true;
is_empty([{_,_}|_]) ->
	false;
is_empty(_) ->
	throw(badarg).

-ifdef(TEST).

em_test_() ->
	RAL = empty(),
	RAL2 = cons(something, RAL),
	[
	?t(RAL =:= []),
	?t(is_empty(RAL) =:= true),
	?t(is_empty(RAL2) =:= false)
	].

empty_test_() ->
	F0 = query_test_fun_gen(
		fun is_empty/1,
		fun ([]) -> true; (_) -> false end),
	general_testing1(F0).
-endif.

head([{_, Tree} | _Rest]) ->
	min(Tree).

-ifdef(TEST).
head_test_() ->
	F0 = query_test_fun_gen(
		fun([]) -> empty; (RAL) -> head(RAL) end,
		fun([]) -> empty; (L) -> hd(L) end),
	general_testing1(F0).
-endif.


% O(log n)
nth(N, RAL) when N > 0 ->
	nth_(N-1, RAL).

nth_(N, [{Size, Tree} | _Rest]) when N < Size ->
	lookup_tree(Tree, N, Size);
nth_(N, [{Size, _Tree} | Rest]) ->
	nth_(N - Size, Rest).

lookup_tree({V, _Left, _Right}, 0, _Size) ->
	V;
lookup_tree({_, Left, Right}, N, Size) ->
	SubSize = Size div 2,
	LorR = (N =< SubSize),
	if
		LorR -> lookup_tree(Left, N - 1, SubSize);
		true -> lookup_tree(Right, N - 1 - SubSize, SubSize)
	end;
lookup_tree({V}, 0, _Size) ->
	V.

tail([{Size, {_V, Left, Right}} | Rest]) ->
	[{Size div 2, Left}, {Size div 2, Right} | Rest];
tail([{1, {_V}} | Rest]) ->
	Rest.

-ifdef(TEST).
tail_test_() ->
	F0 = query_test_fun_gen(
		fun ([]) -> empty; (RAL) -> to_list(tail(RAL)) end,
		fun ([]) -> empty; (L) -> tl(L) end),
	general_testing1(F0).

tail2_test_() ->
	F0 = query_test_fun_gen(
		fun ([]) -> empty; (RAL) -> tail(RAL) end,
		fun ([]) -> empty; (L) -> from_list(tl(L)) end),
	general_testing1(F0).
-endif.


length(RAL) ->
	length(0, RAL).

length(Acc, [{Size, _Tree}|Rest]) ->
	length(Acc+Size, Rest);
length(Acc, []) ->
	Acc.


-ifdef(TEST).
length_test_() ->
	F0 = query_test_fun_gen(
		fun ral:length/1,
		fun erlang:length/1),
	general_testing1(F0).
-endif.



min({V, _Left, _Right}) ->
	V;
min({V}) ->
	V.

max({_, _Left, Right}) ->
	max(Right);
max({V}) ->
	V.

cons(V, [{Size, Tree1}, {Size, Tree2} | Rest]) ->
	[{Size+Size+1, {V, Tree1, Tree2}} | Rest];
cons(V, [ST|Rest]) ->
	[{1, {V}}, ST | Rest];
cons(V, []) ->
	[{1, {V}}].

from_list(NormalList) ->
	lists:foldl(fun(V, RAL) ->
		cons(V, RAL)
	end, empty(), lists:reverse(NormalList)).

from_list_reversed(NormalList) ->
	lists:foldl(fun(V, RAL) ->
		cons(V, RAL)
	end, empty(), NormalList).

% O(n)
to_list(RAL) ->
	foldr(fun(E, Acc) -> [E|Acc] end, [], RAL).

% O(n)
to_list_reversed(RAL) ->
	foldl(fun(E, Acc) -> [E|Acc] end, [], RAL).

-ifdef(TEST).
to_list_test_() ->
	F0 = fun(RAL, NL) ->
		RAL2 = from_list(NL),
		NL2 = to_list(from_list(NL)),
		NL3 = to_list(RAL),
		?assert(NL2 =:= NL),
		?assert(NL2 =:= NL3),
		?assert(RAL2 =:= RAL)
	end,
	general_testing1(F0).
-endif.

% this written explcitly (without lists:foldl usage)

foldl(Fun, Acc1, [{_Size, Tree} | Rest]) ->
	Acc2 = foldl_tree(Fun, Acc1, Tree),
	foldl(Fun, Acc2, Rest);
foldl(_Fun, Acc, []) ->
	Acc.

foldl_tree(Fun, Acc1, {V}) ->
	Fun(V, Acc1);
foldl_tree(Fun, Acc1, {V, Left, Right}) ->
	Acc2 = Fun(V, Acc1),
	Acc3 = foldl_tree(Fun, Acc2, Left),
	Acc4 = foldl_tree(Fun, Acc3, Right),
	Acc4.

foldr(Fun, Acc0, RAL) ->
	lists:foldr(fun({_Size, Tree}, Acc) ->
		foldr_tree(Fun, Acc, Tree)
	end, Acc0, RAL).

foldr_tree(Fun, Acc1, {V}) ->
	Fun(V, Acc1);
foldr_tree(Fun, Acc1, {V, Left, Right}) ->
	Acc2 = foldr_tree(Fun, Acc1, Right),
	Acc3 = foldr_tree(Fun, Acc2, Left),
	Acc4 = Fun(V, Acc3),
	Acc4.

-ifdef(TEST).
folds_test_fun_gen(RAL_FOLD, LIST_FOLD, Fun, Init) ->
	fun(RAL, NL) ->
		R2 = RAL_FOLD(Fun, Init, RAL),
		NR2 = LIST_FOLD(Fun, Init, NL),
		?assert(R2 =:= NR2),
		ok
	end.


fold_fun() ->
	Fun = fun(X, Acc) -> Acc+2*X end,
	Fun.

foldl_test_() ->
	F0 = folds_test_fun_gen(
		fun ral:foldl/3,
		fun lists:foldl/3,
		fold_fun(), 0),
	general_testing1(F0).

foldr_test_() ->
	F0 = folds_test_fun_gen(
		fun ral:foldr/3,
		fun lists:foldr/3,
		fold_fun(), 0),
	general_testing1(F0).
-endif.

foldl_cancelable(Fun, Acc1, [{_Size, Tree} | Rest]) ->
	case foldl_cancelable_tree(Fun, Acc1, Tree) of
		{next, Acc2} ->
			foldl_cancelable(Fun, Acc2, Rest);
		{stop, _} = Last2 ->
			Last2
	end;
foldl_cancelable(_Fun, Acc, []) ->
	Acc.

foldl_cancelable_tree(Fun, Acc1, {V}) ->
	Fun(V, Acc1);
foldl_cancelable_tree(Fun, Acc1, {V, Left, Right}) ->
	case Fun(V, Acc1) of
		{next, Acc2} ->
			case foldl_cancelable_tree(Fun, Acc2, Left) of
				{next, Acc3} ->
					foldl_cancelable_tree(Fun, Acc3, Right);
				{stop, _} = Last3 ->
					Last3
			end;
		{stop, _} = Last2 ->
			Last2
	end.


foreach(Fun, RAL) ->
	lists:foreach(fun({_Size, Tree}) ->
		foreach_tree(Fun, Tree)
	end, RAL).

foreach_tree(Fun, {V}) ->
	Fun(V);
foreach_tree(Fun, {V, Left, Right}) ->
	Fun(V),
	foreach_tree(Fun, Left),
	foreach_tree(Fun, Right).

-ifdef(TEST).
% hot to test foreach?
-endif.

last(RAL) ->
	{_Size, LastTree} = lists:last(RAL),
	max(LastTree).

-ifdef(TEST).
last_test_() ->
	F0 = query_test_fun_gen(
		fun([]) -> empty; (RAL) -> last(RAL) end,
		fun([]) -> empty; (L) -> lists:last(L) end),
	general_testing1(F0).
-endif.

map(Fun, RAL) ->
	lists:map(fun({Size, Tree}) ->
		{Size, map_tree(Fun, Tree)}
	end, RAL).

map_tree(Fun, {V}) ->
	{Fun(V)};
map_tree(Fun, {V, Left, Right}) ->
	NewV = Fun(V),
	NewLeft = map_tree(Fun, Left),
	NewRight = map_tree(Fun, Right),
	{NewV, NewLeft, NewRight}.

-ifdef(TEST).
map_test_fun_gen(RAL_MAP, LIST_MAP, Fun) ->
	fun(RAL, NL) ->
		L2 = to_list(RAL_MAP(Fun, RAL)),
		NL2 = LIST_MAP(Fun, NL),
		?assert(L2 =:= NL2),
		ok
	end.

map_fun() ->
	Fun = fun(X) -> 17*X end,
	Fun.

map_test_() ->
	F0 = map_test_fun_gen(
		fun ral:map/2,
		fun lists:map/2,
		map_fun()),
	general_testing1(F0).
-endif.

mapfoldl(Fun, Acc0, RAL) ->
	lists:mapfoldl(fun({Size, Tree}, Acc) ->
		{NewTree, Acc1} = mapfoldl_tree(Fun, Acc, Tree),
		{{Size, NewTree}, Acc1}
	end, Acc0, RAL).

mapfoldl_tree(Fun, Acc0, {V}) ->
	{NewV, Acc1} = Fun(V, Acc0),
	{{NewV}, Acc1};
mapfoldl_tree(Fun, Acc0, {V, Left, Right}) ->
	{NewV, Acc1} = Fun(V, Acc0),
	{NewLeft, Acc2} = mapfoldl_tree(Fun, Acc1, Left),
	{NewRight, Acc3} = mapfoldl_tree(Fun, Acc2, Right),
	NewTree = {NewV, NewLeft, NewRight},
	{NewTree, Acc3}.

mapfoldr(Fun, Acc0, RAL) ->
	lists:mapfoldr(fun({Size, Tree}, Acc) ->
		{NewTree, Acc1} = mapfoldr_tree(Fun, Acc, Tree),
		{{Size, NewTree}, Acc1}
	end, Acc0, RAL).

mapfoldr_tree(Fun, Acc0, {V}) ->
	{NewV, Acc1} = Fun(V, Acc0),
	{{NewV}, Acc1};
mapfoldr_tree(Fun, Acc0, {V, Left, Right}) ->
	{NewRight, Acc1} = mapfoldr_tree(Fun, Acc0, Right),
	{NewLeft, Acc2} = mapfoldr_tree(Fun, Acc1, Left),
	{NewV, Acc3} = Fun(V, Acc2),
	NewTree = {NewV, NewLeft, NewRight},
	{NewTree, Acc3}.


-ifdef(TEST).
mapfold_fun() ->
	Fun = fun(X, Acc) -> {13*X, Acc+2*X} end,
	Fun.

mapfoldl_test_() ->
	F0 = folds_test_fun_gen(
		fun (Fun, AccIn, RL) ->
			{NewRL, AccOut} = ral:mapfoldl(Fun, AccIn, RL),
			{to_list(NewRL), AccOut}
		end,
		fun lists:mapfoldl/3,
		mapfold_fun(), 0),
	general_testing1(F0).

mapfoldr_test_() ->
	F0 = folds_test_fun_gen(
		fun (Fun, AccIn, RL) ->
			{NewRL, AccOut} = ral:mapfoldr(Fun, AccIn, RL),
			{to_list(NewRL), AccOut}
		end,
		fun lists:mapfoldr/3,
		mapfold_fun(), 0),
	general_testing1(F0).
-endif.


nthtail(0, RAL) ->
	RAL;
nthtail(N, [{Size, _Tree} | Rest]) when N > Size ->
	nthtail(N - Size, Rest);
nthtail(N, [{Size, Tree} | Rest]) ->
	nthtail_tree(N-1, Size, Tree, Rest).

nthtail_tree(N, Size, Tree, Rest) ->
	SubSize = Size div 2,
	if
		N =:= 0 ->
			case Tree of
				{_V, Left, Right} ->
					[{SubSize, Left}, {SubSize, Right} | Rest];
				{_V} ->
					Rest
			end;
		true ->
			{_V, Left, Right} = Tree,
			if
				N > SubSize ->
					nthtail_tree(N - SubSize - 1, SubSize, Right, Rest);
				N =< SubSize ->
					nthtail_tree(N - 1, SubSize, Left, [{SubSize, Right} | Rest])
			end
	end.


-ifdef(TEST).
nthtail_test_() ->
	F0 = fun(RAL, NL, Y) ->
		L2 = try to_list(nthtail(Y, RAL)) catch _:_ -> exception end,
		NL2 = try lists:nthtail(Y, NL) catch _:_ -> exception end,
		?assert(L2 =:= NL2),
		ok
	end,
	general_testing2(F0).
-endif.


dropwhile(Pred, []) when is_function(Pred, 1) ->
	[];
dropwhile(Pred, RAL) ->
	V = head(RAL),
	case Pred(V) of
		true -> dropwhile(Pred, tail(RAL));
		_ -> RAL
	end.

-ifdef(TEST).
dropwhile_test_() ->
	F0 = fun(RAL, NL, Y) ->
		Pred = fun(X) -> X =/= Y end,
		L2 = to_list(dropwhile(Pred, RAL)),
		NL2 = lists:dropwhile(Pred, NL),
		?assert(L2 =:= NL2),
		ok
	end,
	general_testing2(F0).
-endif.


reverse_list([H|T], RAL) ->
	reverse_list(T, cons(H, RAL));
reverse_list([], RAL) ->
	RAL.

-ifdef(TEST).
sample_list() ->
	from_list([a,b,c,d,e,f,g,h,i,j]).

reverse_list_test_() ->
	RAList4 = sample_list(),
	[
		?t([a,b,c,d,e,f,g,h,i,j] =:= to_list(reverse_list([], RAList4))),
		?t([x,a,b,c,d,e,f,g,h,i,j] =:= to_list(reverse_list([x], RAList4))),
		?t([z,y,x,a,b,c,d,e,f,g,h,i,j] =:= to_list(reverse_list([x,y,z], RAList4)))
	].
-endif.


% like nth/2, just add 3,4-rd parameter for new value, and return new trees, instad of value

% O(log n)
replace(N, NewV, RAL) when N > 0 ->
	replace_(N-1, NewV, RAL).

replace_(N, NewV, [{Size, Tree} | Rest]) when N < Size ->
	[{Size, replace_tree(Tree, N, NewV, Size)} | Rest];
replace_(N, NewV, [Keep = {Size, _Tree} | Rest]) ->
	[Keep | replace_(N - Size, NewV, Rest)].

replace_tree({_OldV, Left, Right}, 0, NewV, _Size) ->
	{NewV, Left, Right};
replace_tree({OtherV, Left, Right}, N, NewV, Size) ->
	SubSize = Size div 2,
	LorR = (N =< SubSize),
	if
		LorR -> {OtherV, replace_tree(Left, N - 1, NewV, SubSize), Right};
		true -> {OtherV, Left, replace_tree(Right, N - 1 - SubSize, NewV, SubSize)}
	end;
replace_tree({_OldV}, 0, NewV, _Size) ->
	{NewV}.

-ifdef(TEST).
replace_test_() ->
	RAList4 = sample_list(),
	[
		?t([{1,{x}}] =:= replace(1, x, [{1,{vv}}])),
		?t([x,b,c,d,e,f,g,h,i,j] =:= to_list(replace(1, x, RAList4))),
		?t([a,b,c,d,x,f,g,h,i,j] =:= to_list(replace(5, x, RAList4))),
		?t([a,b,c,d,e,f,g,h,i,x] =:= to_list(replace(10, x, RAList4)))
	].
-endif.
