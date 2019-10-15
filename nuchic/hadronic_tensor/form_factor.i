%module form_factor

%include "std_map.i"
%include "std_string.i"
%include "std_vector.i"

%{
#include "form_factor.hh"
%}

namespace std {
    %template(dvect) vector<double>;
    %template(map_dict) map<string, double>;
}

%rename(__call__) FormFactor::operator();

%include "form_factor.hh"
