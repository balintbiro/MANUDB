// functions for sticky header
window.onscroll = function() {Stick()};

function Stick() {
  var navbar = document.getElementById("navbar");
  var sticky = navbar.offsetTop;
  if (window.pageYOffset > sticky) {
    navbar.classList.add("sticky")
  } else {
    navbar.classList.remove("sticky");
  }
}

// functions for showing/clearing examples
function showExample() {
  if (document.getElementById("inner").value==""){
    let element_list=['>Example_sequence','ACGGGGTACGCGATTTTTTCATCATCAGCTTGTTTTGTCGATCGTACGATCGCTTTAGGGGGGAACAAT']
    for (let i=0, l=element_list.length; i<l; i++)
      document.getElementById("inner").value += element_ist[i]+'\n'
  }
}

function clearExample() {
  document.getElementById("inner").value="";
}

// functions for plotting
var samplePlot = document.getElementById('samplePlot');
function drawLines() {
  var one = {
     x: [1,2,3,4,5,6,7,8,9,10,11],
     y: [1,4,3,3,9,12,10,9,12,20,15],
     mode: 'lines+markers',
     name: 'Sample Data'
  };
  var data = [one];
  var layout = {
    title: {
      text: 'Sample Plot'
    },
    xaxis: {
      title: {text: 'X axis'}
    },
    yaxis: {
      title: {text: 'Y axis'}
    }
  };
  var config = {responsive: true}
  Plotly.newPlot(samplePlot, data, layout, config);
}
