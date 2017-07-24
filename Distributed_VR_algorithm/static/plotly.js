var app = angular.module('myApp', []);
app.controller('myCtrl', function($scope, $http, $interval, $timeout, $q) {
    $scope.iter   = 0;
    $scope.hide_W = true;
    $scope.mu     = 0.8;
    $scope.data_select = {0:true,  1:true,  2:true,  3:true,  4:true,
                          5:false, 6:false, 7:false, 8:false, 9:false}
    $scope.method  = "SVRG";
    $scope.dist_style = "Diffusion"
    $scope.fetched = "Have not fetched data.";
    $scope.cost_value = [];
    $scope.iter_list = [];
    $scope.server_client = "server";
    $scope.one_ip = "127.0.0.1:9999";
    $scope.stop = 0;
    $scope.alreadystop = true;
    $scope.iter_per_call = 5;
    $scope.connect_status = "Not connected";

    $scope.connect =function() {
        $http({
                method : 'POST',
                url: '/connect',
                data: {'server_client': $scope.server_client,
                       'ip':  $scope.one_ip}
        }).then(function mySuccess(response) {
                $scope.connect_status = "Connected";
                // console.log(response.data);
            }, function myError(response){
                $scope.connect_status = "Connected failed";
                console.log("Some Error happened during connected!");
            });
    }
    
    $scope.disconnect =function() {
        $http({
                method : 'GET',
                url: '/disconnect',
        }).then(function mySuccess(response) {
                $scope.connect_status = "Not connected";
                // console.log(response.data);
            }, function myError(response){
                $scope.connect_status = "Disonnected failed";
                console.log("Some Error happened during deconnected!");
            });
    }

    $scope.get_data = function(){
        // stop and reset the running algorithm first
        $scope.stop_alg();
        $scope.rest_alg();
        $scope.fetched="Fetching data now."

        $http({
                method : 'POST',
                url: '/get_data',
                data: {'mask': $scope.data_select}
            }).then(
            function success(response){
                $scope.fetched="Data fetched!";
                $scope.rest_alg();  // used to reset W image
            }, function error(response){
                console.log("Some Error happened during fetched data!");
            })
    }

    $scope.run_alg = function(){
        $scope.hide_W = false;
        $http({
            method : 'POST',
            url: '/run_alg',
            data: {'mu': parseFloat($scope.mu),
                   'method':  $scope.method,
                   'ite': $scope.iter,
                   'iter_per_call': $scope.iter_per_call,
                   'dist_style': $scope.dist_style}
       }).then(
        function success(response){
            $scope.alreadystop = false;
            $scope.hide_W = false;
            var c=document.getElementById("img_tag");
            var port = response.data['running_port'];
            c.src = '/static/visual_W_'+port+'.jpg?random='+new Date().getTime(); // refresh image
            if (response.data['cost_value'] != 'skipped') {
                $scope.cost_value.push(response.data['cost_value'])
                $scope.iter_list.push($scope.iter)
                // console.log($scope.cost_value)
                plot_cost($scope.iter_list, $scope.cost_value)
            }
            $scope.iter = $scope.iter + parseInt($scope.iter_per_call);
            // console.log($scope.stop)
            if ($scope.stop === 0) {
                $timeout(function(){$scope.run_alg();},200);
                // console.log("enter run branch at iter " + $scope.iter)
            } else {
                $scope.alreadystop = true;
                $scope.stop = 0; // re-define it so that next time it will keep running
                // console.log("enter stop branch")
            }
        }, function error(response){
            console.log("Some Error happened during run algorithm!");
            $scope.stop_alg();
            $scope.hide_W = true;
            $scope.iter = 0;
            $scope.cost_value = [];
            $scope.iter_list = [];
        })  
    }

    $scope.stop_alg = function(){
        if (!$scope.alreadystop) {
            $scope.stop = 1;
        }
    }

    $scope.rest_alg = function(){
        $http({
                method : 'GET',
                url: '/rest_alg'
            }).then(
            function success(response){
                var c=document.getElementById("img_tag");
                var port = response.data['running_port'];
                if (!angular.isDefined(port)){
                    c.src = '/static/visual_W_'+port+'.jpg?random='+new Date().getTime();
                }
                $scope.hide_W = true;
                $scope.iter = 0;
                $scope.cost_value = [];
                $scope.iter_list = [];
            }, function error(response){
                console.log("Some Error happened during reset plot!");
            })
    }

    // $scope.$watch($scope.cost_value, function() {
    //     console.log($scope.cost_value)
        
    // });
});

app.config(function($interpolateProvider) {
  $interpolateProvider.startSymbol('{[{');
  $interpolateProvider.endSymbol('}]}');
});

plot_func = function(data){

    var x_pos = [];
    var y_pos = [];
    var node_indx = [];
    for (var key in data.node) {
        x_pos.push(data.node[key].pos[0]);
        y_pos.push(data.node[key].pos[1]);
        node_indx.push(key)
    }
    console.log("x_pos:"+x_pos)
    console.log("y_pos:"+x_pos)

    trace = {
        x: x_pos,
        y: y_pos,
        text: node_indx,
        textposition: "bottom center",
        mode: 'markers+text'
    };

    layout = {
        margin: {t:0},
        xaxis: {showline: false,
                showgrid: false,
                zeroline: false,
                ticks: '',
                showticklabels: false},
        yaxis: {showline: false,
                showgrid: false,
                zeroline: false,
                ticks: '',
                showticklabels: false}
    };

    place = document.getElementById('plot-holder');
    Plotly.plot( place, [trace], layout);
}

plot_cost = function(iters, data){
    if (data == []) {
        return 
    }
    trace = {
        x: iters,
        y: data,
    };
    layout = {margin: {t:0, l:25, r:10},
              xaxis: {showline: true,
                      zeroline: false
                     },
              yaxis: {showline: true,
                      zeroline: false
                     }
            }
    place = document.getElementById('plot-holder');
    Plotly.newPlot( place, [trace], layout);
}