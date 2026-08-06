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
	$p_1_plus_mo = ( 1 - (0.8)**$n );
	$p_1_plus_fr = ( 1 - (0.8)**($n-1) );
	$p_1_plus_we = ( 1 - (0.8)**($n-2) );
	$p_zero_we = 1. - $p_1_plus_we;
	$p_1_plus_tu_and_th = ( 1 - (0.75)**($n-2) ) * ( 1 - (0.75)**($n-3) ); # 0.75 because zero on wednesday
	$p_zero_we_but_1_plus_tu_and_th = $p_zero_we * $p_1_plus_tu_and_th;
	$p_1_plus_mo_and_fr = $p_1_plus_mo  * $p_1_plus_fr;
	$final = $p_1_plus_mo * $p_1_plus_fr * ( $p_1_plus_we + $p_zero_we_but_1_plus_tu_and_th) ;
	#MC
	$n_ident = 0;
	$n_mf_ident = 0;
	for (1..$n_try)
	{
		@f = ( 0, 0, 0, 0, 0 );
		for $qq ( 1..$n )
		{
			$q = int rand 5;
			++$f[$q];
		}
		$ident = 0; 
		if ( $f[0] && $f[4] && ( $f[2] || ( $f[1] && $f[3] ) ) )
		{
			$ident = 1;
			$n_ident += $ident ;
		}
		$mf_ident = 0; 
		if ( $f[0] && $f[4] )
		{
			$mf_ident = 1;
			$n_mf_ident += $mf_ident ;
		}
		#print + join "_", @f; print " ",  $ident ? "alI" : "alN", "\t", $mf_ident ? 'mfI' : 'mfN', "\n";
	}
	if ($debug)
	{
		print $n, "\t",  $p_1_plus_mo,  "\t",  $p_1_plus_fr, "\t",  $p_1_plus_mo_and_fr, "\t", $final, "\tMC: ", $n_ident / $n_try,"\tMF: ", $n_mf_ident / $n_try,  "\t", $n_ident - $n_mf_ident, "\n";
	}
	else
	{
		print $n, "\t", $final, "\t", $n_ident / $n_try,"\n";
	}
}
