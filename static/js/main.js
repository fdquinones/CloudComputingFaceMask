
(function ($) {
	"use strict";
	$('.column100').on('mouseover',function(){
		var table1 = $(this).parent().parent().parent();
		var table2 = $(this).parent().parent();
		var verTable = $(table1).data('vertable')+"";
		var column = $(this).data('column') + ""; 

		$(table2).find("."+column).addClass('hov-column-'+ verTable);
		$(table1).find(".row100.head ."+column).addClass('hov-column-head-'+ verTable);
	});

	$('.column100').on('mouseout',function(){
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
		content += '<tr class="row100">';
            content += '<td class="column100 column1" data-column="column1">' + new Date(val.fecha).toLocaleString() + '</td>';
            content += '<td class="column100 column2" data-column="column2">' + val.cpu + ' mb</td>';
            content += '<td class="column100 column3" data-column="column3">' + val.imagen + '</td>';
        content += '</tr>';
		$('#tableData tbody').append(content);
		//addCommentElement(postElement, data.key, data.val().text, data.val().author);
	});
	
	// Attach an asynchronous callback to read the data at our posts reference
	eventsRef.on('value', (snapshot) => {
	  console.log(snapshot.val());
	}, (errorObject) => {
	  console.log('The read failed: ' + errorObject.name);
	});
	
	var database = firebase.database();
	database.ref('facemask').once('value', function(snapshot) {
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