#!/usr/bin/perl 
#===============================================================================
#
#         FILE: day_of_week.pl

$n_try = 100000;

$debug = 1;
$debug = 0;

if ($debug)
{
	print "\#N", "\t",  "p_1_plus_mo",  "\t",  "p_1_plus_fr", "\t",  "p_1_plus_mo_and_fr", "\t", "FINAL", "\tMC: ", "n_ident/n_try","\tMF: ", "n_mf_ident/n_try",  "\t", "n_ident-n_mf_ident", "\n";
}
else
{
	print "\#N", "\t", "FINAL", "\t", "MC\n";
}

for $n (1..250)
{
	$p_1_plus_mo = ( 1 - (5./6.)**$n );
	$p_1_plus_tu = ( 1 - (5./6.)**($n-1) );
	$p_1_plus_we = ( 1 - (5./6.)**($n-2) );
	$p_1_plus_th = ( 1 - (5./6.)**($n-3) );
	$p_1_plus_fr = ( 1 - (5./6.)**($n-4) );
	$p_1_plus_sa = ( 1 - (5./6.)**($n-5) );
	
	$final = $p_1_plus_mo * $p_1_plus_tu * $p_1_plus_we * $p_1_plus_th * $p_1_plus_fr * $p_1_plus_sa ; 

	#MC
	$n_ident = 0;
	$n_mf_ident = 0;
	for (1..$n_try)
	{
		@f = ( 0, 0, 0, 0, 0, 0 );
		for $qq ( 1..$n )
		{
			$q = int rand 6;
			++$f[$q];
		}
		$ident = 0; 
		if ( $f[0] && $f[1] && $f[2] && $f[3] && $f[4] && $f[5] )
		{
			$ident = 1;
			$n_ident += $ident ;
		}
	}
	if ($debug)
	{
		print $n, "\t", $final, "\tMC: ", $n_ident / $n_try, "\n";
	}
	else
	{
		print $n, "\t", $final, "\t", $n_ident / $n_try,"\n";
	}
}
