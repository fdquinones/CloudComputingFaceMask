
(function ($) {
	"use strict";
	var url = "wss://covidutpl.tk/camera";
	var canvas = document.getElementById('video-canvas');
	var player = new JSMpeg.Player(url, {canvas: canvas});
	//player.play();

	$('#tableData').on('mouseover', '.column100', function(){
		var table1 = $(this).parent().parent().parent();
		var table2 = $(this).parent().parent();
		var verTable = $(table1).data('vertable')+"";
		var column = $(this).data('column') + ""; 

		$(table2).find("."+column).addClass('hov-column-'+ verTable);
		$(table1).find(".row100.head ."+column).addClass('hov-column-head-'+ verTable);
	});

	//cambiar la imagen dinamicamente
	$('#tableData').on('mouseover', '.row100.infoRow', function(){
		var pathImage = $(this).closest('tr').children('td.data').text();
		$("#imgDetection").attr("src","static/" + pathImage);
	});

	$('#tableData').on('mouseout', '.column100',function(){
		var table1 = $(this).parent().parent().parent();
		var table2 = $(this).parent().parent();
		var verTable = $(table1).data('vertable')+"";
		var column = $(this).data('column') + ""; 

		$(table2).find("."+column).removeClass('hov-column-'+ verTable);
		$(table1).find(".row100.head ."+column).removeClass('hov-column-head-'+ verTable);
	});
	
	// Your web app's Firebase configuration
	var firebaseConfig = {
		apiKey: "AIzaSyBNXxTi201phes4B24moqj_bUIAZvo8cbw",
		authDomain: "utplcovid.firebaseapp.com",
		projectId: "utplcovid",
		storageBucket: "utplcovid.appspot.com",
		messagingSenderId: "1052404064289",
		appId: "1:1052404064289:web:385ce48af2131cc38ec360"
	};
	// Initialize Firebase
	firebase.initializeApp(firebaseConfig);

	/*var article = {
		'fecha': Date.now(),
		'cpu': 50,
		'imagen': 'hola.jpg',
	}

	firebase.database().ref('facemask/' + Date.now()).set(article);*/
	
	var eventsRef = firebase.database().ref('facemask');
	eventsRef.on('child_added', (data) => {
		console.log("llegando evento en portal de utpl");
		var content = '';
		console.log(data.val());
		console.log(data.key);
		var val = data.val();

		if(val.label){
			content += '<tr class="row100 infoRow">';
            content += '<td class="column100 column1" data-column="column1">' + new Date(val.fecha).toLocaleString() + '</td>';
            content += '<td class="column100 column2" data-column="column2">' + val.cpuP + '</td>';
			content += '<td class="column100 column3" data-column="column3">' + val.virtualMemoryT + '</td>';
			content += '<td class="column100 column4" data-column="column4">' + val.virtualMemory + ' </td>';
			content += '<td class="column100 column5" data-column="column5">' + val.virtualMemoryP + ' </td>';
			content += '<td class="column100 column6" data-column="column6">' + val.label + ' </td>';
			content += '<td class="column100 column7" data-column="column7">' + val.prediction.toFixed(2) + ' </td>';
            content += '<td class="column100 column8 data" data-column="column8">' + val.imagen + '</td>';
        	content += '</tr>';
			$('#tableData tbody tr:first').before(content);
		}
	});
	
	// Attach an asynchronous callback to read the data at our posts reference
	eventsRef.on('value', (snapshot) => {
	  console.log('hola   .. .');
	  console.log(snapshot.val());
	}, (errorObject) => {
	  console.log('The read failed: ' + errorObject.name);
	});
	
	var database = firebase.database();
	database.ref('facemask').once('value', function(snapshot) {
		console.log('hello   .. .');
        if (snapshot.exists()) {
          var content = '';
          snapshot.forEach(function(data) {
            var val = data.val();
            content += '<tr>';
            content += '<td>' + val.firstName + '</td>';
            content += '<td>' + val.lastName + '</td>';
            content += '<td>' + val.cin + '</td>';
            content += '<td>' + val.numTel + '</td>';
            content += '<td>' + val.pw + '</td>';
            content += '</tr>';
          });
          $('#ex-table').append(content);
        }
      });



    

})(jQuery);