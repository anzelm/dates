#!/usr/bin/perl 
#===============================================================================
#
#         FILE: clinics_closed.pl
#
#        USAGE: ./clinics_closed.pl  
#
#SPDX-License-Identifier: MIT

$week_days = 261;
$clinics_closed = 10;

my $n ; #procedures

print "N", "\t", "exclude_1",  "\t", "exclude_51", "\t", "exclude_50+", "\t", "exclude_49+", "\t", "exclude_48+", "\n";
for $n (1..150)
{
	# One specific week (or day; but day of week can be cracked easily) not being excluded by one specific procedure
	$p_one_shift_not_into_clinics_closed = ($week_days - $clinics_closed) / $week_days ;   # 251/261   
	# One specific week NOT excluded by N independent procedures
	$p_n_shift_not_into_clinics_closed = ( $p_one_shift_not_into_clinics_closed ) ** $n;   #  ( 251/261 )^N   
	# this allows a week: (week NOT excluded by N procedures)
	$allow_one_week = $p_n_shift_not_into_clinics_closed;
	# this excludes 52 weeks:
	$exclude_51_weeks = ( 1 - $allow_one_week ) ** 51;
	$exclude_50_weeks = $exclude_51_weeks +  $allow_one_week * ( 1 - $allow_one_week ) ** 50 * 51; 
	$exclude_49_weeks = $exclude_50_weeks + $allow_one_week ** 2 * ( 1 - $allow_one_week ) ** 49 * 51 * 50 / 2; 
	$exclude_48_weeks = $exclude_49_weeks + $allow_one_week ** 3 * ( 1 - $allow_one_week ) ** 48 * 51 * 50 * 49 / ( 2 * 3 ); 
	# this excludes a week:
	$p_n_shift_into_clinics_closed = 1 - $p_n_shift_not_into_clinics_closed;
	print $n, "\t", $p_n_shift_into_clinics_closed, "\t", $exclude_51_weeks, "\t", $exclude_50_weeks, "\t", $exclude_49_weeks, "\t", $exclude_48_weeks, "\n";
}
